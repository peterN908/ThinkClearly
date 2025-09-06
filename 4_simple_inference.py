"""
Simple inference for CLEAR-aware Janus-4o fine-tune.

What this does:
- Loads a base Janus-4o model and applies LoRA adapters saved by training, plus the CLEAR head.
- Encodes a user-provided input image to tokens.
- Builds a SFT-style prompt with two image placeholders and the instruction.
- Runs a greedy CLEAR-aware token generation loop:
  * At each step, map LM hidden to logits via gen_head_clear (last logit is CLEAR).
  * If CLEAR is chosen, the emitted token index refers back to the input token at that position.
  * Otherwise, the chosen token id is used.
- Decodes generated tokens to an image and writes out:
  * raw_model.png: direct decode of generated tokens
  * token_composite.png: image decoded from tokens after copying CLEAR positions from input tokens
  * pixel_composite.png: pixel-level composite; CLEAR positions copy pixels from the input image
  * clear_mask.png: visualization of CLEAR positions

Notes:
- Training saved LoRA adapters via save_pretrained(output_dir), tokenizer, and gen_head_clear.pt + clear_meta.json.
- We expect 576 image tokens (24x24) for 384x384 images with 16px patches.
"""

import os
import json
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage, ImageFilter

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


IMAGE_SIZE = 384
PATCH_SIZE = 16
TOKENS_PER_IMAGE = (IMAGE_SIZE // PATCH_SIZE) * (IMAGE_SIZE // PATCH_SIZE)  # 24*24 = 576


class GenHeadWithClear(nn.Module):
    """
    Must mirror the training-time head: base gen_head + extra CLEAR logit at the end.
    """
    def __init__(self, base_head: nn.Module):
        super().__init__()
        self.base_head = base_head
        self.clear_linear: Optional[nn.Linear] = None

    def ensure_initialized(self, hidden_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.clear_linear is None:
            self.clear_linear = nn.Linear(hidden_size, 1, bias=True)
            self.clear_linear.to(device=device, dtype=dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base_logits = self.base_head(hidden)
        if self.clear_linear is None:
            in_dim = hidden.shape[-1]
            self.ensure_initialized(in_dim, hidden.device, hidden.dtype)
        clear_logits = self.clear_linear(hidden)
        return torch.cat([base_logits, clear_logits], dim=-1)


@dataclass
class VLChatProcessorOutput:
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def select_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def load_model_and_processor(model_path: str) -> Tuple[VLChatProcessor, MultiModalityCausalLM, torch.device, torch.dtype]:
    device, dtype = select_device_and_dtype()
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    load_kwargs = {"trust_remote_code": True}
    if device.type == "cuda":
        # Use correct key expected by HF
        load_kwargs["torch_dtype"] = dtype
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    vl_gpt = vl_gpt.to(device).eval()
    return vl_chat_processor, vl_gpt, device, dtype


def attach_clear_head(vl_gpt: MultiModalityCausalLM, device: torch.device, dtype: torch.dtype, clear_head_state_path: str | None) -> None:
    if not hasattr(vl_gpt, "gen_head_clear"):
        base_head = vl_gpt.gen_head  # type: ignore[attr-defined]
        vl_gpt.gen_head_clear = GenHeadWithClear(base_head)
    # Ensure init and load weights if available
    emb = vl_gpt.language_model.get_input_embeddings()
    hidden_size = int(getattr(emb, "embedding_dim", emb.weight.shape[1]))
    vl_gpt.gen_head_clear.ensure_initialized(hidden_size, device, dtype)  # type: ignore[attr-defined]
    if clear_head_state_path and os.path.exists(clear_head_state_path):
        state = torch.load(clear_head_state_path, map_location="cpu")
        vl_gpt.gen_head_clear.load_state_dict(state, strict=False)  # type: ignore[attr-defined]
        try:
            w = vl_gpt.gen_head_clear.clear_linear.weight  # type: ignore[attr-defined]
            b = vl_gpt.gen_head_clear.clear_linear.bias  # type: ignore[attr-defined]
            print(f"Loaded CLEAR head: weight_norm={float(w.norm().cpu()):.4f}, bias_mean={float(b.mean().cpu()):.4f}")
        except Exception:
            pass


def maybe_load_lora_adapters(vl_gpt: MultiModalityCausalLM, adapters_dir: str):
    """
    Loading LoRA adapters saved via save_pretrained(output_dir).
    If present, PEFT will be required in runtime environment.
    """
    adapter_config = os.path.join(adapters_dir, "adapter_config.json")
    if os.path.exists(adapter_config):
        from peft import PeftModel
        print(f"Loading LoRA adapters from {adapters_dir}")
        model_with_lora = PeftModel.from_pretrained(vl_gpt, adapters_dir)
        print("LoRA adapters loaded successfully")
        return model_with_lora
    else:
        print(f"No LoRA adapter config found at {adapter_config}, using base model")
    return vl_gpt


def build_sft_prefix_ids(instruction: str, vl_chat_processor: VLChatProcessor) -> torch.LongTensor:
    input_img_tokens = (
        vl_chat_processor.image_start_tag
        + vl_chat_processor.image_tag * vl_chat_processor.num_image_tokens
        + vl_chat_processor.image_end_tag
        + vl_chat_processor.image_start_tag
        + vl_chat_processor.pad_tag * vl_chat_processor.num_image_tokens
        + vl_chat_processor.image_end_tag
    )
    prompts = input_img_tokens + instruction
    conversation = [
        {"role": "<|User|>", "content": prompts},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    sft_format = sft_format + vl_chat_processor.image_start_tag
    input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
    return input_ids


def insert_input_image_embeds(tokens: torch.LongTensor, inputs_embeds: torch.Tensor, input_image_tokens: torch.Tensor, vl_gpt: MultiModalityCausalLM, vl_chat_processor: VLChatProcessor) -> torch.Tensor:
    with torch.no_grad():
        image_embeds_input = vl_gpt.prepare_gen_img_embeds(input_image_tokens)
    seq = tokens[0].tolist()
    try:
        start_id = vl_chat_processor.image_start_id  # type: ignore[attr-defined]
    except Exception:
        start_id = vl_chat_processor.tokenizer.encode(vl_chat_processor.image_start_tag)[-1]
    start_positions = [i for i, tid in enumerate(seq) if tid == start_id]
    if len(start_positions) >= 2:
        second_start = start_positions[1]
        offset = second_start + 1
    elif len(start_positions) == 1:
        offset = start_positions[0] + 1
    else:
        offset = max(0, inputs_embeds.shape[1] - image_embeds_input.shape[1])
    end_offset = offset + image_embeds_input.shape[1]
    if end_offset <= inputs_embeds.shape[1]:
        inputs_embeds[:, offset:end_offset, :] = image_embeds_input
    else:
        pad_len = end_offset - inputs_embeds.shape[1]
        pad = torch.zeros((1, pad_len, inputs_embeds.shape[-1]), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, pad], dim=1)
        inputs_embeds[:, offset:offset + image_embeds_input.shape[1], :] = image_embeds_input
    return inputs_embeds


def encode_image_to_tokens(image: PILImage.Image, vl_chat_processor: VLChatProcessor, vl_gpt: MultiModalityCausalLM, device: torch.device, dtype: torch.dtype) -> List[int]:
    inputs = vl_chat_processor.image_processor([image.convert("RGB")], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    if device.type == "cuda":
        pixel_values = pixel_values.to(dtype)
    _, _, info = vl_gpt.gen_vision_model.encode(pixel_values)
    tokens: torch.Tensor = info[2].detach().reshape(pixel_values.shape[0], -1)
    return [int(t) for t in tokens.squeeze(0).to("cpu").tolist()]


def decode_tokens_to_image(tokens: torch.Tensor, vl_gpt: MultiModalityCausalLM) -> np.ndarray:
    parallel_size = tokens.shape[0]
    dec = vl_gpt.gen_vision_model.decode_code(tokens.to(dtype=torch.int), shape=[parallel_size, 8, IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return dec


def map_clear_to_source(pred_tokens: torch.Tensor, src_tokens: torch.Tensor, clear_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replace CLEAR ids in pred_tokens with corresponding ids from src_tokens.
    Returns (mapped_tokens, clear_mask_bool).
    """
    is_clear = pred_tokens == int(clear_id)
    mapped = torch.where(is_clear, src_tokens.to(pred_tokens.device), pred_tokens)
    return mapped, is_clear


def _median_filter_bool(mask_grid: np.ndarray, k: int = 3, thresh: Optional[int] = None, passes: int = 1) -> np.ndarray:
    """Simple median-like filter for boolean grids using edge padding.
    k must be odd. Returns a boolean grid of the same shape.
    """
    assert k % 2 == 1 and k >= 1
    h, w = mask_grid.shape
    pad = k // 2
    out = mask_grid.astype(np.uint8)
    for _ in range(max(1, int(passes))):
        acc = np.zeros_like(out, dtype=np.int32)
        padded = np.pad(out, pad, mode="edge")
        for dy in range(-pad, pad + 1):
            ys = pad + dy
            ye = ys + h
            for dx in range(-pad, pad + 1):
                xs = pad + dx
                xe = xs + w
                acc += padded[ys:ye, xs:xe]
        thr = (k * k) // 2 + 1 if thresh is None else int(thresh)
        out = (acc >= thr).astype(np.uint8)
    return out.astype(bool)


def visualize_and_save(
    input_img: PILImage.Image,
    raw_decoded: np.ndarray,
    token_composite_decoded: np.ndarray,
    clear_mask_1d: np.ndarray,
    out_dir: str,
    mask_smooth_ksize: int = 3,
    mask_smooth_passes: int = 1,
    feather_radius_px: int = 8,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Save raw model image
    PILImage.fromarray(raw_decoded[0]).save(os.path.join(out_dir, "raw_model.png"))

    # Save token-level composite image
    PILImage.fromarray(token_composite_decoded[0]).save(os.path.join(out_dir, "token_composite.png"))

    # CLEAR mask visualization (24x24 upscaled to full res)
    mask_grid_bool = clear_mask_1d.reshape(IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE).astype(bool)
    # Save the raw, unsmoothed mask
    mask_raw_img = PILImage.fromarray((mask_grid_bool.astype(np.uint8) * 255), mode="L").resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.NEAREST)
    mask_raw_img.save(os.path.join(out_dir, "clear_mask.png"))

    # Optional token-grid smoothing to reduce 1-patch jaggies before feathering
    if mask_smooth_ksize and mask_smooth_ksize > 1:
        mask_grid_bool = _median_filter_bool(mask_grid_bool, k=int(mask_smooth_ksize), passes=int(max(1, mask_smooth_passes)))

    # Feather edges in pixel space using a Gaussian blur to create soft alpha
    mask_img_smooth = PILImage.fromarray((mask_grid_bool.astype(np.uint8) * 255), mode="L").resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.NEAREST)
    if feather_radius_px and feather_radius_px > 0:
        mask_alpha = mask_img_smooth.filter(ImageFilter.GaussianBlur(radius=int(feather_radius_px)))
    else:
        mask_alpha = mask_img_smooth
    mask_alpha_np = np.asarray(mask_alpha, dtype=np.float32) / 255.0

    # Pixel-level composite: copy pixels from input where CLEAR
    base = np.array(input_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BICUBIC), dtype=np.uint8)
    edited = np.array(token_composite_decoded[0], dtype=np.uint8)
    # Soft alpha blend instead of hard copy to hide seams
    alpha = mask_alpha_np[..., None].clip(0.0, 1.0)
    pixel_comp_f = alpha * base.astype(np.float32) + (1.0 - alpha) * edited.astype(np.float32)
    pixel_comp = np.clip(pixel_comp_f, 0, 255).astype(np.uint8)
    PILImage.fromarray(pixel_comp).save(os.path.join(out_dir, "pixel_composite.png"))


def generate_tokens_with_clear(
    vl_gpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    device: torch.device,
    dtype: torch.dtype,
    instruction: str,
    input_image_tokens: List[int],
    log_every: int = 24,
    topk: int = 5,
    trace_file: Optional[str] = None,
    clear_bias: float = 0.0,
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, int]:
    """
    Greedy CLEAR-aware generation with optional per-step logging:
    - Build SFT prefix with two image placeholders
    - Insert input image embeds in the second placeholder
    - At each step, map hidden -> logits via gen_head_clear
      * last index = CLEAR id; if chosen, copy input token at this position
      * else, use the sampled id as output token
    - If trace_file is provided, write one JSONL record per step with raw id, effective id,
      CLEAR decision, source token, and top-k candidates.
    Returns (generated_tokens [1,576], clear_mask [576 bool], generated_raw [1,576], clear_id).
    """
    if not hasattr(vl_gpt, "gen_head_clear"):
        raise RuntimeError("CLEAR head missing; cannot generate with CLEAR tokens. Make sure to load gen_head_clear.pt.")

    input_ids = build_sft_prefix_ids(instruction, vl_chat_processor).to(device).unsqueeze(0)
    embed_tokens = vl_gpt.language_model.get_input_embeddings()
    inputs_embeds = embed_tokens(input_ids)
    input_image_tokens_t = torch.as_tensor(input_image_tokens, dtype=torch.long, device=device).unsqueeze(0)
    inputs_embeds = insert_input_image_embeds(input_ids, inputs_embeds, input_image_tokens_t, vl_gpt, vl_chat_processor)

    generated = torch.zeros((1, TOKENS_PER_IMAGE), dtype=torch.long, device=device)
    generated_raw = torch.zeros((1, TOKENS_PER_IMAGE), dtype=torch.long, device=device)
    clear_mask = np.zeros((TOKENS_PER_IMAGE,), dtype=np.bool_)

    past_key_values = None
    base_vocab = None

    # Ensure directory exists for trace file, if requested
    if trace_file is not None:
        trace_dir = os.path.dirname(trace_file)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)

    def _step_body(i: int, logits_tensor: torch.Tensor, base_vocab_val: int, tf_handle) -> None:
        nonlocal inputs_embeds
        # Apply CLEAR bias to the CLEAR logit (last index)
        if clear_bias != 0.0:
            logits_tensor = logits_tensor.clone()
            logits_tensor[0, 0, base_vocab_val] = logits_tensor[0, 0, base_vocab_val] + float(clear_bias)
        # Greedy next id
        next_id_local = int(torch.argmax(logits_tensor[0, 0], dim=-1).item())
        generated_raw[0, i] = next_id_local
        if next_id_local == base_vocab_val:
            tok_local = int(input_image_tokens[i])
            clear_mask[i] = True
            is_clear_local = True
        else:
            tok_local = next_id_local
            is_clear_local = False
        generated[0, i] = tok_local

        # CLEAR logit and rank
        logits_vec = logits_tensor[0, 0]
        clear_logit_val = float(logits_vec[base_vocab_val].detach().to("cpu").item())
        # rank 1 means highest
        clear_rank_val = int((logits_vec > logits_vec[base_vocab_val]).sum().detach().to("cpu").item()) + 1

        # Optional top-k logging
        k = int(min(topk, logits_tensor.shape[-1])) if topk and topk > 0 else 0
        topk_ids_out = []
        topk_scores_out = []
        if k > 0:
            topk_scores, topk_ids = torch.topk(logits_tensor[0, 0], k=k)
            topk_ids_out = [int(x) for x in topk_ids.detach().to("cpu").tolist()]
            topk_scores_out = [float(x) for x in topk_scores.detach().to("cpu").tolist()]

        if (i % max(1, log_every) == 0) or (i == TOKENS_PER_IMAGE - 1):
            preview_pairs = list(zip(topk_ids_out, [round(s, 2) for s in topk_scores_out]))
            print(f"Step {i+1}/{TOKENS_PER_IMAGE}: raw={next_id_local}{' (CLEAR)' if is_clear_local else ''} -> eff={tok_local}; clear_logit={clear_logit_val:.2f} rank={clear_rank_val}; top{k}: {preview_pairs}")

        if tf_handle is not None:
            rec = {
                "step": int(i),
                "raw_id": int(next_id_local),
                "effective_id": int(tok_local),
                "src_id": int(input_image_tokens[i]),
                "is_clear": bool(is_clear_local),
                "copied_from_src": bool(is_clear_local),
                "clear_id": int(base_vocab_val),
                "clear_logit": clear_logit_val,
                "clear_rank": int(clear_rank_val),
                "topk_ids": topk_ids_out,
                "topk_scores": topk_scores_out,
                "clear_bias": float(clear_bias),
            }
            tf_handle.write(json.dumps(rec) + "\n")

        # Feed next image token embedding
        img_embeds_local = vl_gpt.prepare_gen_img_embeds(generated[:, i])
        inputs_embeds = img_embeds_local.unsqueeze(1)

    if trace_file is not None:
        with open(trace_file, "w", encoding="utf-8") as tf:
            for i in range(TOKENS_PER_IMAGE):
                outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                hidden_last = outputs.last_hidden_state[:, -1:, :]  # [1,1,H]

                logits = vl_gpt.gen_head_clear(hidden_last)  # type: ignore[attr-defined]
                if base_vocab is None:
                    base_vocab = int(logits.shape[-1] - 1)

                _step_body(i, logits, int(base_vocab), tf)
    else:
        for i in range(TOKENS_PER_IMAGE):
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            hidden_last = outputs.last_hidden_state[:, -1:, :]  # [1,1,H]

            logits = vl_gpt.gen_head_clear(hidden_last)  # type: ignore[attr-defined]
            if base_vocab is None:
                base_vocab = int(logits.shape[-1] - 1)

            _step_body(i, logits, int(base_vocab), None)

    return generated.to("cpu"), clear_mask, generated_raw.to("cpu"), int(base_vocab)


def generate_tokens_with_clear_cfg(
    vl_gpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    device: torch.device,
    dtype: torch.dtype,
    instruction: str,
    input_image: PILImage.Image,
    log_every: int = 24,
    topk: int = 5,
    trace_file: Optional[str] = None,
    temperature: float = 1.0,
    parallel_size: int = 1,
    cfg_weight: float = 5.0,
    cfg_weight2: float = 5.0,
    clear_bias: float = 0.0,
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, int]:
    """
    Classifier-Free Guidance (CFG) generation adapted from the reference implementation
    with CLEAR-aware head and logging. Uses tripled batch: [cond_full, cond_part, uncond].
    """
    if not hasattr(vl_gpt, "gen_head_clear"):
        raise RuntimeError("CLEAR head missing; cannot generate with CLEAR tokens. Make sure to load gen_head_clear.pt.")

    # Build SFT format string
    input_img_tokens = (
        vl_chat_processor.image_start_tag
        + vl_chat_processor.image_tag * vl_chat_processor.num_image_tokens
        + vl_chat_processor.image_end_tag
        + vl_chat_processor.image_start_tag
        + vl_chat_processor.pad_tag * vl_chat_processor.num_image_tokens
        + vl_chat_processor.image_end_tag
    )
    prompts = input_img_tokens + instruction
    conversation = [
        {"role": "<|User|>", "content": prompts},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    sft_format = sft_format + vl_chat_processor.image_start_tag

    # Encode input image to tokens and embeds
    inputs = vl_chat_processor.image_processor([input_image.convert("RGB")], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    if device.type == "cuda":
        pixel_values = pixel_values.to(dtype)
    _, _, info_input = vl_gpt.gen_vision_model.encode(pixel_values)
    image_tokens_input = info_input[2].detach().reshape(pixel_values.shape[0], -1)
    image_embeds_input = vl_gpt.prepare_gen_img_embeds(image_tokens_input)

    input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))

    # Prepare tripled tokens for CFG
    tokens = torch.zeros((parallel_size * 3, len(input_ids)), dtype=torch.long)
    pre_data: List[VLChatProcessorOutput] = []
    img_len = 1
    for i in range(parallel_size * 3):
        tokens[i, :] = input_ids
        if i % 3 == 2:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
            encoder_pixel_values = pixel_values
            pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=encoder_pixel_values, input_ids=tokens[i-2], num_image_tokens=torch.IntTensor([vl_chat_processor.num_image_tokens] * img_len)))
            pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=encoder_pixel_values, input_ids=tokens[i-1], num_image_tokens=torch.IntTensor([vl_chat_processor.num_image_tokens] * img_len)))
            pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=None, input_ids=tokens[i], num_image_tokens=torch.IntTensor([])))

    prepare_inputs = vl_chat_processor.batchify(pre_data)
    inputs_embeds = vl_gpt.prepare_inputs_embeds(
        input_ids=tokens.to(device),
        pixel_values=prepare_inputs['pixel_values'].to(dtype).to(device) if prepare_inputs['pixel_values'] is not None else None,
        images_emb_mask=prepare_inputs['images_emb_mask'].to(device),
        images_seq_mask=prepare_inputs['images_seq_mask'].to(device),
    )

    # Insert the input image embeds into second placeholder for cond rows
    image_gen_indices = (tokens == vl_chat_processor.image_end_id).nonzero()
    for ii, ind in enumerate(image_gen_indices):
        if ii % 4 == 0:
            offset = ind[1] + 2
            inputs_embeds[ind[0], offset: offset + image_embeds_input.shape[1], :] = image_embeds_input[(ii // 2) % img_len]

    generated = torch.zeros((parallel_size, TOKENS_PER_IMAGE), dtype=torch.long, device=device)
    generated_raw = torch.zeros((parallel_size, TOKENS_PER_IMAGE), dtype=torch.long, device=device)
    clear_mask = np.zeros((TOKENS_PER_IMAGE,), dtype=np.bool_)

    past_key_values = None
    base_vocab = None

    if trace_file is not None:
        trace_dir = os.path.dirname(trace_file)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        tfh = open(trace_file, "w", encoding="utf-8")
    else:
        tfh = None

    try:
        for i in range(TOKENS_PER_IMAGE):
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state  # [B, T, H]

            logits_all = vl_gpt.gen_head_clear(hidden_states[:, -1:, :])  # [B, 1, V+1]
            if base_vocab is None:
                base_vocab = int(logits_all.shape[-1] - 1)

            # Split CFG triplets
            logits_full = logits_all[0::3, 0, :]
            logits_part = logits_all[1::3, 0, :]
            logits_uncond = logits_all[2::3, 0, :]

            logits_cond = (logits_full + cfg_weight2 * logits_part) / (1.0 + cfg_weight2)
            logits = logits_uncond + cfg_weight * (logits_cond - logits_uncond)
            # Apply CLEAR bias on combined logits (last index)
            if clear_bias != 0.0:
                logits[:, base_vocab] = logits[:, base_vocab] + float(clear_bias)

            probs = torch.softmax(logits / max(1e-6, float(temperature)), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [parallel_size, 1]
            next_ids = next_token.squeeze(dim=-1)  # [parallel_size]

            # Record only the first sample for clear mask/debug
            rid = int(next_ids[0].item())
            generated_raw[0, i] = rid
            if rid == base_vocab:
                tok_eff = int(image_tokens_input[0, i].item())
                clear_mask[i] = True
                is_clear = True
            else:
                tok_eff = rid
                is_clear = False

            generated[0, i] = tok_eff

            # CLEAR logit and rank from the combined logits (first parallel)
            logits_vec = logits[0]
            clear_logit_val = float(logits_vec[base_vocab].detach().to("cpu").item())
            clear_rank_val = int((logits_vec > logits_vec[base_vocab]).sum().detach().to("cpu").item()) + 1

            # Optional top-k
            k = int(min(topk, logits_vec.shape[-1])) if topk and topk > 0 else 0
            topk_ids_out = []
            topk_scores_out = []
            if k > 0:
                topk_scores, topk_ids = torch.topk(logits_vec, k=k)
                topk_ids_out = [int(x) for x in topk_ids.detach().to("cpu").tolist()]
                topk_scores_out = [float(x) for x in topk_scores.detach().to("cpu").tolist()]

            if (i % max(1, log_every) == 0) or (i == TOKENS_PER_IMAGE - 1):
                preview_pairs = list(zip(topk_ids_out, [round(s, 2) for s in topk_scores_out]))
                print(f"Step {i+1}/{TOKENS_PER_IMAGE} [CFG]: raw={rid}{' (CLEAR)' if is_clear else ''} -> eff={tok_eff}; clear_logit={clear_logit_val:.2f} rank={clear_rank_val}; top{k}: {preview_pairs}")

            if tfh is not None:
                rec = {
                    "step": int(i),
                    "raw_id": int(rid),
                    "effective_id": int(tok_eff),
                    "src_id": int(image_tokens_input[0, i].item()),
                    "is_clear": bool(is_clear),
                    "copied_from_src": bool(is_clear),
                    "clear_id": int(base_vocab),
                    "clear_logit": clear_logit_val,
                    "clear_rank": int(clear_rank_val),
                    "topk_ids": topk_ids_out,
                    "topk_scores": topk_scores_out,
                    "cfg_weight": float(cfg_weight),
                    "cfg_weight2": float(cfg_weight2),
                    "temperature": float(temperature),
                    "clear_bias": float(clear_bias),
                }
                tfh.write(json.dumps(rec) + "\n")

            # Prepare next-step embeds for the tripled batch.
            # IMPORTANT: Map CLEAR sentinel to source token id before embedding to avoid OOB.
            next_eff = next_token.clone()  # [parallel_size, 1]
            if rid == base_vocab:
                # Replace with source token id for this step
                next_eff[0, 0] = image_tokens_input[0, i].to(dtype=next_eff.dtype, device=next_eff.device)
            next_token_trip = torch.cat([next_eff, next_eff, next_eff], dim=1).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token_trip)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
    finally:
        if tfh is not None:
            tfh.close()

    return generated.to("cpu"), clear_mask, generated_raw.to("cpu"), int(base_vocab)


def run_clear_inference(
    input_image: Union[str, PILImage.Image] = "test_0.png",
    instruction: str = "Make a small red sticker on the top right.",
    base_model: str = "FreedomIntelligence/Janus-4o-7B",
    finetune_dir: str = "/workspace/sft_janus4o_clear/",
    out_dir: str = "./inference_outputs",
    use_cfg: bool = True,
    temperature: float = 1.0,
    parallel_size: int = 1,
    cfg_weight: float = 5.0,
    cfg_weight2: float = 5.0,
    clear_bias: float = 0.0,
    # Anti-blockiness options
    mask_smooth_ksize: int = 3,
    mask_smooth_passes: int = 1,
    feather_radius_px: int = 8,
) -> dict:
    """
    Notebook-friendly entry point. Returns a dict with outputs and paths.

    Usage in Jupyter:
        from 4_simple_inference import run_clear_inference
        out = run_clear_inference(
            input_image="images/test_1.png",
            instruction="Add a small red sticker on the top right.",
            base_model="FreedomIntelligence/Janus-4o-7B",
            finetune_dir="/workspace/sft_janus4o_clear/",
            out_dir="./inference_outputs",
        )
    """
    vl_chat_processor, vl_gpt, device, dtype = load_model_and_processor(base_model)
    vl_gpt = maybe_load_lora_adapters(vl_gpt, finetune_dir)

    clear_meta = os.path.join(finetune_dir, "clear_meta.json")
    clear_head_state = os.path.join(finetune_dir, "gen_head_clear.pt")
    if os.path.exists(clear_meta):
        with open(clear_meta, "r", encoding="utf-8") as f:
            _ = json.load(f)
    attach_clear_head(vl_gpt, device, dtype, clear_head_state)

    # Load/normalize input image
    if isinstance(input_image, PILImage.Image):
        input_img = input_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BICUBIC)
    else:
        input_img = PILImage.open(str(input_image)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BICUBIC)

    input_tokens = encode_image_to_tokens(input_img, vl_chat_processor, vl_gpt, device, dtype)

    # Ensure output directory exists early for trace writing
    os.makedirs(out_dir, exist_ok=True)

    # Per-step JSONL token trace
    trace_file = os.path.join(out_dir, "token_trace.jsonl")

    # Generate CLEAR-aware tokens
    if use_cfg:
        gen_tokens_cpu, clear_mask, gen_tokens_raw, clear_id = generate_tokens_with_clear_cfg(
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            device=device,
            dtype=dtype,
            instruction=instruction,
            input_image=input_img,
            log_every=24,
            topk=5,
            trace_file=trace_file,
            temperature=temperature,
            parallel_size=parallel_size,
            cfg_weight=cfg_weight,
            cfg_weight2=cfg_weight2,
            clear_bias=clear_bias,
        )
    else:
        gen_tokens_cpu, clear_mask, gen_tokens_raw, clear_id = generate_tokens_with_clear(
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            device=device,
            dtype=dtype,
            instruction=instruction,
            input_image_tokens=input_tokens,
            log_every=24,
            topk=5,
            trace_file=trace_file,
            clear_bias=clear_bias,
        )

    # Decode raw generated tokens
    raw_decoded = decode_tokens_to_image(gen_tokens_cpu, vl_gpt)

    # Token-level composite: map CLEAR -> source before decoding
    input_tokens_t = torch.as_tensor(input_tokens, dtype=torch.long).unsqueeze(0)
    token_composite, _ = map_clear_to_source(gen_tokens_raw, input_tokens_t, clear_id)
    token_composite_decoded = decode_tokens_to_image(token_composite, vl_gpt)

    # Visualize and save
    visualize_and_save(
        input_img=input_img,
        raw_decoded=raw_decoded,
        token_composite_decoded=token_composite_decoded,
        clear_mask_1d=clear_mask,
        out_dir=out_dir,
        mask_smooth_ksize=mask_smooth_ksize,
        mask_smooth_passes=mask_smooth_passes,
        feather_radius_px=feather_radius_px,
    )

    # Also save a sanity decode of the input tokens
    input_decoded = decode_tokens_to_image(torch.as_tensor(input_tokens, dtype=torch.long).unsqueeze(0), vl_gpt)
    PILImage.fromarray(input_decoded[0]).save(os.path.join(out_dir, "input_decoded.png"))

    print(f"Saved outputs to {out_dir}:")
    print("- raw_model.png")
    print("- token_composite.png")
    print("- clear_mask.png")
    print("- pixel_composite.png")
    print(f"CLEAR tokens predicted: {int(clear_mask.sum())}/{TOKENS_PER_IMAGE} ({100.0*float(clear_mask.sum())/float(TOKENS_PER_IMAGE):.2f}%)")
    print(f"CLEAR sentinel id (in raw sequence): {clear_id}")
    print(f"Token trace written to: {trace_file}")

    # Pretty view of raw tokens with <CLEAR> markers for inspection in notebooks
    raw_list = gen_tokens_raw.squeeze(0).tolist()
    pretty = ["<CLEAR>" if int(t) == int(clear_id) else int(t) for t in raw_list]
    eff_list = gen_tokens_cpu.squeeze(0).tolist()
    # Print a compact preview of tokens
    preview_n = 48
    print(f"First {preview_n} raw ids: {raw_list[:preview_n]}")
    print(f"First {preview_n} effective ids: {eff_list[:preview_n]}")

    return {
        "generated_tokens": gen_tokens_cpu,
        "generated_tokens_raw": gen_tokens_raw,
        "generated_tokens_pretty": pretty,
        "clear_id": clear_id,
        "clear_mask": clear_mask,
        "raw_decoded": raw_decoded,
        "token_composite_decoded": token_composite_decoded,
        "out_dir": out_dir,
        "paths": {
            "raw_model": os.path.join(out_dir, "raw_model.png"),
            "token_composite": os.path.join(out_dir, "token_composite.png"),
            "clear_mask": os.path.join(out_dir, "clear_mask.png"),
            "pixel_composite": os.path.join(out_dir, "pixel_composite.png"),
        },
        "trace_path": trace_file,
    }


# Sample inference
# out = run_clear_inference(
#   input_image="test_0.png",
#   instruction="Make the sea shell red.",
#   base_model="FreedomIntelligence/Janus-4o-7B",
#   finetune_dir="/workspace/sft_janus4o_clear/",
#   out_dir="inference_outputs",
#   use_cfg=True,
#   temperature=0.8,
#   parallel_size=1,
#   cfg_weight=5.0,
#   cfg_weight2=5.0,
#   clear_bias=5.0,
# )

# from IPython.display import Image
# Image(filename='inference_outputs/pixel_composite.png') 
