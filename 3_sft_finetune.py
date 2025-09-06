"""
SFT fine-tuning (LoRA) to teach Janus-4o a CLEAR class for image editing.

What we implement (minimal, codebook-preserving):
- Expand only the image gen head to K+1 logits (K real codes + 1 CLEAR), via a wrapper
  `GenHeadWithClear(base_head)` that concatenates a learned scalar logit for CLEAR.
  CLEAR_ID is the last index (K). The VQ codebook/tokenizer remain unchanged.

- Data/labels (from 2_prepare_data.py):
  Each record has `instruction`, `input_tokens` (source codes), and `output_tokens`, where
  unchanged positions are the literal string "<CLEAR>" and changed positions are ints.
  We build:
    * `teacher_tokens`: ints for changed positions; `input_tokens[i]` for CLEAR positions
    * `labels`: ints for changed positions; placeholder -1 for CLEAR positions
    * `clear_mask`: True at CLEAR positions

- Training loop: vectorized teacher-forcing
  1) Build an SFT prompt with two image placeholders; insert the input image embeddings
     into the second placeholder region.
  2) For the L image steps, feed embeddings where CLEAR positions take the corresponding
     input-token embedding and others take the target-token embedding.
  3) Map hidden → logits with `gen_head_clear` (K+1). Build final targets by setting
     `labels_final[clear_mask] = CLEAR_ID` and leaving other targets as the true K-way ids.
  4) Compute class-weighted CE with a reduced weight for CLEAR (e.g., 0.5) to avoid collapse.
  5) Freeze base weights; apply LoRA to LM + base gen head; train the CLEAR classifier linear
     directly. Save LoRA adapters and `gen_head_clear.pt` for inference.

- Inference (see 4_simple_inference.py):
  Generate raw tokens (may contain CLEAR_ID), then call `map_clear_to_source` to replace
  CLEAR with source token ids before calling the decoder. Optionally composite pixels using
  the CLEAR mask for perfect-preserve regions.
"""


import os
import random
import json
from typing import List, Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from peft import LoraConfig, TaskType, get_peft_model


# HuggingFace cache setup (consistent with other scripts)
os.environ["HF_HOME"] = "/workspace"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hub"
os.environ["HF_HUB_CACHE"] = "/workspace/hub"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)


def select_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def load_model_and_processor(model_path: str) -> Tuple[VLChatProcessor, MultiModalityCausalLM, torch.device, torch.dtype]:
    device, dtype = select_device_and_dtype()

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device.type == "cuda":
        load_kwargs["torch_dtype"] = dtype

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs,
    )

    vl_gpt = vl_gpt.to(device).eval()
    return vl_chat_processor, vl_gpt, device, dtype


def quick_supervision_stats(jsonl_path: str, max_samples: int = 0) -> Tuple[int, int, float]:
    sup = 0
    tot = 0
    seen = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            rec = json.loads(line)
            out = rec.get("output_tokens", [])
            tot += len(out)
            sup += sum(1 for t in out if isinstance(t, int))
            seen += 1
    ratio = sup / max(1, tot)
    print(f"Supervision check -> samples: {seen}, supervised tokens: {sup}, ratio: {ratio:.2%} @ {jsonl_path}")
    return sup, tot, ratio


class PreparedMagicBrushDataset(Dataset):
    def __init__(self, jsonl_path: str, limit: int = -1):
        self.records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit > 0 and i >= limit:
                    break
                rec = json.loads(line)
                # Expect keys: instruction, input_tokens, output_tokens
                self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        instruction: str = rec["instruction"]
        input_tokens: List[int] = rec["input_tokens"]
        output_tokens_raw: List[Any] = rec["output_tokens"]

        # Build teacher-forcing tokens, labels (ints for changed; placeholder for CLEAR), and a CLEAR mask
        teacher_tokens: List[int] = []
        labels: List[int] = []
        clear_mask: List[int] = []
        int_count = 0
        clear_count = 0
        
        for i, out_tok in enumerate(output_tokens_raw):
            is_int = isinstance(out_tok, int)
            # DEBUG: Also check for other numeric types
            is_numeric = isinstance(out_tok, (int, float)) and not isinstance(out_tok, str)
            
            if idx < 3 and i < 5:  # Log first few tokens of first few samples
                print(f"  Token {i}: {out_tok} (type: {type(out_tok)}, isinstance(int): {is_int}, is_numeric: {is_numeric})")
            
            if is_int:
                teacher_tokens.append(int(out_tok))
                labels.append(int(out_tok))
                clear_mask.append(0)
                int_count += 1
            else:
                # "<CLEAR>" -> feed input token embedding; model should predict CLEAR id
                teacher_tokens.append(int(input_tokens[i]))
                labels.append(-1)  # placeholder; will be replaced with CLEAR_ID during loss
                clear_mask.append(1)
                clear_count += 1
                

        return {
            "instruction": instruction,
            "input_tokens": input_tokens,
            "teacher_tokens": teacher_tokens,
            "labels": labels,
            "clear_mask": clear_mask,
        }


def build_sft_prefix_ids(instruction: str, vl_chat_processor: VLChatProcessor) -> torch.LongTensor:
    """
    Construct the SFT-format conversational prompt with:
    - User message containing two image placeholders (input + pad region)
    - Assistant starts an output image with image_start_tag
    Returns tokenized ids as LongTensor [seq_len].
    """
    # Two placeholders: [input image] + [pad image], then instruction
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
    # Assistant starts generating an output image
    sft_format = sft_format + vl_chat_processor.image_start_tag

    input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
    return input_ids


def insert_input_image_embeds(
    tokens: torch.LongTensor,
    inputs_embeds: torch.Tensor,
    input_image_tokens: torch.Tensor,
    vl_gpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
) -> torch.Tensor:
    """
    Insert the input image embeddings into the second image placeholder (the pad region).
    tokens: [1, T]
    inputs_embeds: [1, T, H]
    input_image_tokens: [1, 576]
    Returns modified inputs_embeds.
    """
    with torch.no_grad():
        image_embeds_input = vl_gpt.prepare_gen_img_embeds(input_image_tokens)

    # Locate the second image_start position and fill the following num_image_tokens slots
    seq = tokens[0].tolist()
    try:
        start_id = vl_chat_processor.image_start_id  # type: ignore[attr-defined]
    except Exception:
        # Fallback: derive image_start_id by encoding the tag alone
        start_id = vl_chat_processor.tokenizer.encode(vl_chat_processor.image_start_tag)[-1]

    start_positions = [i for i, tid in enumerate(seq) if tid == start_id]
    if len(start_positions) >= 2:
        second_start = start_positions[1]
        offset = second_start + 1
    elif len(start_positions) == 1:
        offset = start_positions[0] + 1
    else:
        # If not found, place at end minus space for embeds
        offset = max(0, inputs_embeds.shape[1] - image_embeds_input.shape[1])

    end_offset = offset + image_embeds_input.shape[1]
    if end_offset <= inputs_embeds.shape[1]:
        inputs_embeds[:, offset:end_offset, :] = image_embeds_input
    else:
        # If not enough room, pad by concatenation
        pad_len = end_offset - inputs_embeds.shape[1]
        pad = torch.zeros((1, pad_len, inputs_embeds.shape[-1]), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, pad], dim=1)
        inputs_embeds[:, offset:offset + image_embeds_input.shape[1], :] = image_embeds_input

    return inputs_embeds


def compute_vectorized_gen_loss(
    vl_gpt: MultiModalityCausalLM,
    prefix_inputs_embeds: torch.Tensor,
    teacher_tokens: torch.Tensor,
    labels: torch.Tensor,
    input_tokens: torch.Tensor,
    clear_mask: torch.Tensor,
    clear_weight: float = 0.5,
) -> torch.Tensor:
    """
    Vectorized teacher-forcing: feed prefix + gold token embeddings, predict each gold token.
    - CLEAR positions are teacher-forced with the corresponding input token embedding
      and trained to predict the dedicated CLEAR class.
    """
    device = prefix_inputs_embeds.device
    # Prepare embeddings for teacher tokens and for input tokens (for CLEAR)
    gold_embeds_teacher = vl_gpt.prepare_gen_img_embeds(teacher_tokens)  # [B, L, H]
    gold_embeds_input = vl_gpt.prepare_gen_img_embeds(input_tokens)      # [B, L, H]
    # Mix: CLEAR positions take input embeddings; others take teacher token embeddings
    cm = clear_mask.to(dtype=gold_embeds_teacher.dtype).unsqueeze(-1)  # [B, L, 1]
    gold_embeds = gold_embeds_teacher * (1.0 - cm) + gold_embeds_input * cm
    # Full input sequence to the LM
    full_inputs = torch.cat([prefix_inputs_embeds, gold_embeds], dim=1)  # [B, T+L, H]
    outputs = vl_gpt.language_model.model(inputs_embeds=full_inputs)
    hidden = outputs.last_hidden_state  # [B, T+L, H]

    B = hidden.shape[0]
    prefix_len = prefix_inputs_embeds.shape[1]
    L = teacher_tokens.shape[1]

    # Positions used to predict image tokens:
    # - position (prefix_len-1) predicts teacher_tokens[:, 0]
    # - positions prefix_len-1 + i predict teacher_tokens[:, i] for i>=1
    # Gather the slice starting at prefix_len-1 with length L
    start = prefix_len - 1
    pred_hidden = hidden[:, start:start + L, :]  # [B, L, H]

    # Map to code logits via (augmented) gen head with an extra CLEAR class
    if not hasattr(vl_gpt, "gen_head_clear"):
        raise RuntimeError("gen_head_clear not attached; CLEAR token not available.")
    logits = vl_gpt.gen_head_clear(pred_hidden)
    base_vocab = logits.shape[-1] - 1
    clear_id = base_vocab  # last index is CLEAR
    
    # DEBUG: Log loss computation details for first call
    if not hasattr(compute_vectorized_gen_loss, "_debug_logged"):
        print(f"DEBUG: Loss computation - logits shape: {logits.shape}")
        print(f"DEBUG: base_vocab: {base_vocab}, clear_id: {clear_id}")
        print(f"DEBUG: clear_mask shape: {clear_mask.shape}")
        print(f"DEBUG: clear_mask sum: {clear_mask.sum().item()}")
        print(f"DEBUG: labels shape: {labels.shape}")
        print(f"DEBUG: labels before final: {labels[0][:10].tolist()}")
        compute_vectorized_gen_loss._debug_logged = True
    
    # Compute loss in float32 for numerical stability
    logits = logits.float()

    # Build final targets: changed -> original labels; CLEAR -> clear_id
    labels_final = labels.clone()
    labels_final[clear_mask] = int(clear_id)
    
    # DEBUG: Continue logging for first call
    if hasattr(compute_vectorized_gen_loss, "_debug_logged") and compute_vectorized_gen_loss._debug_logged:
        print(f"DEBUG: labels_final after clear assignment: {labels_final[0][:10].tolist()}")
        print(f"DEBUG: Number of CLEAR labels: {(labels_final == clear_id).sum().item()}")
        print(f"DEBUG: Number of non-CLEAR labels: {(labels_final != clear_id).sum().item()}")
        compute_vectorized_gen_loss._debug_logged = False  # Only log once

    # Class-weighted CE: down-weight CLEAR to reduce collapse
    num_classes = int(logits.shape[-1])
    class_weights = torch.ones((num_classes,), device=logits.device, dtype=torch.float32)
    # last index is CLEAR; make it configurable
    class_weights[-1] = float(clear_weight)
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(
        logits.reshape(B * L, -1),
        labels_final.reshape(B * L).to(device),
    )
    return loss


class GenHeadWithClear(nn.Module):
    """
    Wraps the base gen_head to add an extra CLEAR logit at the end of the vocabulary.
    The CLEAR logit is produced by a lazily-initialized Linear(hidden, 1).
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
        base_logits = self.base_head(hidden)  # [B, L, V]
        if self.clear_linear is None:
            in_dim = hidden.shape[-1]
            self.ensure_initialized(in_dim, hidden.device, hidden.dtype)
        clear_logits = self.clear_linear(hidden)  # [B, L, 1]
        return torch.cat([base_logits, clear_logits], dim=-1)


def apply_lora_adapters(
    vl_gpt: MultiModalityCausalLM,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> MultiModalityCausalLM:
    """
    Wrap the model with PEFT LoRA adapters. Dynamically selects only supported inner
    Linear modules across the LM (q/k/v/o, mlp) and within gen_head, avoiding container
    modules that PEFT cannot wrap.
    """
    # Base suffix patterns we want across the LM stack
    lm_suffixes = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # Collect exact module names to target: only Linear submodules
    target_names: List[str] = []
    for name, module in vl_gpt.named_modules():
        if isinstance(module, nn.Linear):
            # LM transformer projections by suffix
            if any(name.endswith(sfx) for sfx in lm_suffixes):
                target_names.append(name)
                continue
            # Any Linear inside gen_head hierarchy
            if name.startswith("gen_head"):
                # Skip the CLEAR classifier linear; train it directly (not via LoRA)
                if name.endswith("clear_linear"):
                    continue
                target_names.append(name)

    if not target_names:
        # Fallback: at least cover LM projections by suffix in case names differ
        target_names = lm_suffixes

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_names,
    )

    vl_gpt = get_peft_model(vl_gpt, config)
    return vl_gpt


def train_sft(
    jsonl_path: str = "/workspace/prepared_magicbrush.jsonl",
    model_path: str = "FreedomIntelligence/Janus-4o-7B",
    output_dir: str = "/workspace/sft_janus4o_clear/",
    epochs: int = 1,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    limit: int = -1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    clear_weight: float = 0.5,
    val_jsonl_path: Optional[str] = None,
    val_limit: int = -1,
    validate_every: int = 1,
    validate_every_steps: int = -1,
    val_count: int = 50,
    split_seed: int = 42,
    auto_tune_clear_weight: bool = False,
    clear_weight_eta: float = 0.5,
    clear_weight_min: float = 0.1,
    clear_weight_max: float = 1.5,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    vl_chat_processor, vl_gpt, device, dtype = load_model_and_processor(model_path)
    # Attach augmented gen head for CLEAR token prediction
    if not hasattr(vl_gpt, "gen_head_clear"):
        print("DEBUG: Attaching GenHeadWithClear to model")
        base_head = vl_gpt.gen_head  # type: ignore[attr-defined]
        vl_gpt.gen_head_clear = GenHeadWithClear(base_head)
    else:
        print("DEBUG: gen_head_clear already exists on model")
    # Sanity check on provided dataset
    quick_supervision_stats(jsonl_path)
    # Freeze base; apply LoRA to
    # LM + gen_head (excluding CLEAR classifier)
    for p in vl_gpt.parameters():
        p.requires_grad = False
    vl_gpt = apply_lora_adapters(
        vl_gpt,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    # Instantiate CLEAR classifier and mark trainable
    trainable_params: List[torch.nn.Parameter] = []
    if hasattr(vl_gpt, "gen_head_clear") and isinstance(vl_gpt.gen_head_clear, GenHeadWithClear):
        # Derive hidden size from token embeddings
        emb = vl_gpt.language_model.get_input_embeddings()
        if hasattr(emb, "embedding_dim"):
            hidden_size = int(getattr(emb, "embedding_dim"))  # type: ignore[arg-type]
        else:
            hidden_size = int(emb.weight.shape[1])
        vl_gpt.gen_head_clear.ensure_initialized(hidden_size, device, dtype)
        for p in vl_gpt.gen_head_clear.clear_linear.parameters():
            p.requires_grad = True
            trainable_params.append(p)
    # Collect all trainable params: LoRA + CLEAR classifier
    existing_param_ids = {id(p) for p in trainable_params}
    trainable_params.extend([p for p in vl_gpt.parameters() if p.requires_grad and id(p) not in existing_param_ids])
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in vl_gpt.parameters())
    print(f"LoRA active. Trainable params: {total_trainable:,} / {total_params:,} ({100.0*total_trainable/max(1,total_params):.2f}%)")

    def custom_collate(batch):
        """Custom collate function to avoid default tensor conversion."""
        return {
            'instruction': [item['instruction'] for item in batch],
            'input_tokens': [item['input_tokens'] for item in batch],
            'teacher_tokens': [item['teacher_tokens'] for item in batch],
            'labels': [item['labels'] for item in batch],
            'clear_mask': [item['clear_mask'] for item in batch],
        }
    
    # Build train/val split
    train_json_path = jsonl_path
    val_json_path = val_jsonl_path
    if val_json_path is None:
        # Perform a simple random split: ~val_count for validation, rest for training
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        n_total = len(lines)
        k_val = max(1, min(int(val_count), n_total))
        idxs = list(range(n_total))
        rng = random.Random(int(split_seed))
        rng.shuffle(idxs)
        val_idxs = set(idxs[:k_val])
        train_lines = [ln for i, ln in enumerate(lines) if i not in val_idxs]
        val_lines = [ln for i, ln in enumerate(lines) if i in val_idxs]
        split_dir = os.path.join(output_dir, "splits")
        os.makedirs(split_dir, exist_ok=True)
        train_json_path = os.path.join(split_dir, "train.jsonl")
        val_json_path = os.path.join(split_dir, "val.jsonl")
        with open(train_json_path, "w", encoding="utf-8") as tf:
            tf.writelines(train_lines)
        with open(val_json_path, "w", encoding="utf-8") as vf:
            vf.writelines(val_lines)
        print(f"Random split (seed={split_seed}): total={n_total} -> train={len(train_lines)}, val={len(val_lines)} (target val_count={val_count})")

    # Create loaders from split files
    dataset = PreparedMagicBrushDataset(jsonl_path=train_json_path, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    print(f"Loaded training dataset: {len(dataset)} samples from {train_json_path}")

    val_dataset = PreparedMagicBrushDataset(jsonl_path=val_json_path, limit=val_limit)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    print(f"Loaded validation dataset: {len(val_dataset)} samples from {val_json_path}")

    optim = torch.optim.AdamW(trainable_params, lr=learning_rate)

    vl_gpt.train()

    # Track a mutable copy of clear_weight for optional on-the-fly tuning
    current_clear_weight: float = float(clear_weight)

    def _save_checkpoint(epoch_idx: int, step_idx: int) -> None:
        """Save a periodic checkpoint with LoRA adapters, tokenizer, and CLEAR head."""
        ckpt_dir = os.path.join(output_dir, "checkpoints", f"epoch_{epoch_idx:03d}_step_{step_idx:06d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Preserve current mode and switch to eval for consistent saves
        was_training = vl_gpt.training
        vl_gpt.eval()
        # Save LoRA adapters (PEFT)
        vl_gpt.save_pretrained(ckpt_dir)
        # Save tokenizer for portability
        try:
            vl_chat_processor.tokenizer.save_pretrained(ckpt_dir)
        except Exception:
            pass
        # Save CLEAR head state and simple metadata
        meta = {}
        if hasattr(vl_gpt, "gen_head_clear"):
            try:
                torch.save(vl_gpt.gen_head_clear.state_dict(), os.path.join(ckpt_dir, "gen_head_clear.pt"))
                meta["has_clear_head"] = True
            except Exception:
                pass
        try:
            with open(os.path.join(ckpt_dir, "clear_meta.json"), "w", encoding="utf-8") as mf:
                json.dump(meta, mf)
        except Exception:
            pass
        # Restore previous mode
        if was_training:
            vl_gpt.train()
    
    def _run_validation(epoch_idx: int, step_idx: Optional[int] = None) -> Tuple[float, float, float]:
        """Run a validation pass.
        Returns (avg_loss, actual_clear_ratio, predicted_clear_ratio) and prints a summary line.
        """
        vl_gpt.eval()
        with torch.no_grad():
            val_total_loss = 0.0
            val_steps = 0
            val_supervised = 0
            val_clear = 0
            val_pred_clear = 0
            val_tokens_total = 0
            for vbatch in val_loader:
                instruction_v: str = vbatch["instruction"][0]
                input_tokens_list_v: List[int] = vbatch["input_tokens"][0]
                teacher_tokens_list_v: List[int] = vbatch["teacher_tokens"][0]
                labels_list_v: List[int] = vbatch["labels"][0]
                clear_mask_list_v: List[int] = vbatch["clear_mask"][0]

                input_tokens_v = torch.as_tensor(input_tokens_list_v, dtype=torch.long, device=device).unsqueeze(0)
                teacher_tokens_v = torch.as_tensor(teacher_tokens_list_v, dtype=torch.long, device=device).unsqueeze(0)
                labels_v = torch.as_tensor(labels_list_v, dtype=torch.long, device=device).unsqueeze(0)
                clear_mask_v = torch.as_tensor(clear_mask_list_v, dtype=torch.bool, device=device).unsqueeze(0)

                val_supervised += int((~clear_mask_v).sum().item())
                val_clear += int(clear_mask_v.sum().item())
                val_steps += 1

                prefix_ids_v = build_sft_prefix_ids(instruction_v, vl_chat_processor).to(device).unsqueeze(0)
                embed_tokens_v = vl_gpt.language_model.get_input_embeddings()
                prefix_embeds_v = embed_tokens_v(prefix_ids_v)
                prefix_embeds_v = insert_input_image_embeds(
                    tokens=prefix_ids_v,
                    inputs_embeds=prefix_embeds_v,
                    input_image_tokens=input_tokens_v,
                    vl_gpt=vl_gpt,
                    vl_chat_processor=vl_chat_processor,
                )

                # Compute logits and loss; also count predicted CLEAR
                with torch.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else torch.cpu.amp.autocast():  # type: ignore[attr-defined]
                    # Mirror compute_vectorized_gen_loss to access logits
                    gold_embeds_teacher_v = vl_gpt.prepare_gen_img_embeds(teacher_tokens_v)
                    gold_embeds_input_v = vl_gpt.prepare_gen_img_embeds(input_tokens_v)
                    cm_v = clear_mask_v.to(dtype=gold_embeds_teacher_v.dtype).unsqueeze(-1)
                    gold_embeds_v = gold_embeds_teacher_v * (1.0 - cm_v) + gold_embeds_input_v * cm_v
                    full_inputs_v = torch.cat([prefix_embeds_v, gold_embeds_v], dim=1)
                    outputs_v = vl_gpt.language_model.model(inputs_embeds=full_inputs_v)
                    hidden_v = outputs_v.last_hidden_state
                    prefix_len_v = prefix_embeds_v.shape[1]
                    L_v = teacher_tokens_v.shape[1]
                    start_v = prefix_len_v - 1
                    pred_hidden_v = hidden_v[:, start_v:start_v + L_v, :]
                    logits_v = vl_gpt.gen_head_clear(pred_hidden_v)
                    base_vocab_v = logits_v.shape[-1] - 1
                    clear_id_v = base_vocab_v
                    # Predicted CLEAR count across positions
                    preds_v = torch.argmax(logits_v, dim=-1)
                    val_pred_clear += int((preds_v == int(clear_id_v)).sum().detach().cpu().item())
                    val_tokens_total += int(L_v)
                    # Build weighted CE loss (same as training)
                    logits_f = logits_v.float()
                    labels_final_v = labels_v.clone()
                    labels_final_v[clear_mask_v] = int(clear_id_v)
                    num_classes_v = int(logits_f.shape[-1])
                    class_weights_v = torch.ones((num_classes_v,), device=logits_f.device, dtype=torch.float32)
                    class_weights_v[-1] = float(current_clear_weight)
                    loss_fct_v = nn.CrossEntropyLoss(weight=class_weights_v)
                    vloss = loss_fct_v(logits_f.reshape(-1, num_classes_v), labels_final_v.reshape(-1).to(logits_f.device))
                    val_total_loss += float(vloss.detach().cpu().item())

            val_avg = val_total_loss / max(1, val_steps)
            val_clr_ratio = val_clear / max(1, (val_clear + val_supervised))
            val_pred_ratio = (float(val_pred_clear) / float(max(1, val_tokens_total)))
            if step_idx is None:
                print(f"Validation after epoch {epoch_idx}: avg loss: {val_avg:.4f}, CLEAR actual: {val_clr_ratio:.2%}, CLEAR predicted: {val_pred_ratio:.2%}")
            else:
                print(f"Validation at epoch {epoch_idx}, step {step_idx}: avg loss: {val_avg:.4f}, CLEAR actual: {val_clr_ratio:.2%}, CLEAR predicted: {val_pred_ratio:.2%}")
        vl_gpt.train()
        return val_avg, val_clr_ratio, val_pred_ratio
    for epoch in range(epochs):
        total_loss = 0.0
        total_supervised = 0
        total_clear = 0
        step = 0
        for batch in loader:
            # Train with batch_size=1 for clarity and to avoid complex prefix padding
            instruction: str = batch["instruction"][0]
            input_tokens_list: List[int] = batch["input_tokens"][0]
            teacher_tokens_list: List[int] = batch["teacher_tokens"][0]
            labels_list: List[int] = batch["labels"][0]
            clear_mask_list: List[int] = batch["clear_mask"][0]
            

            # Convert to tensors
            input_tokens = torch.as_tensor(input_tokens_list, dtype=torch.long, device=device).unsqueeze(0)
            teacher_tokens = torch.as_tensor(teacher_tokens_list, dtype=torch.long, device=device).unsqueeze(0)
            labels = torch.as_tensor(labels_list, dtype=torch.long, device=device).unsqueeze(0)
            clear_mask = torch.as_tensor(clear_mask_list, dtype=torch.bool, device=device).unsqueeze(0)

            supervised_count = int((~clear_mask).sum().item())
            clear_count = int(clear_mask.sum().item())
            
            
            total_supervised += supervised_count
            total_clear += clear_count
            step += 1

            # Build text prefix and base embeds
            prefix_ids = build_sft_prefix_ids(instruction, vl_chat_processor).to(device).unsqueeze(0)
            embed_tokens = vl_gpt.language_model.get_input_embeddings()
            prefix_inputs_embeds = embed_tokens(prefix_ids)

            # Insert input image embeds into the second placeholder region
            prefix_inputs_embeds = insert_input_image_embeds(
                tokens=prefix_ids,
                inputs_embeds=prefix_inputs_embeds,
                input_image_tokens=input_tokens,
                vl_gpt=vl_gpt,
                vl_chat_processor=vl_chat_processor,
            )

            # Mixed precision on CUDA bfloat16
            with torch.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else torch.cpu.amp.autocast():  # type: ignore[attr-defined]
                loss = compute_vectorized_gen_loss(
                    vl_gpt=vl_gpt,
                    prefix_inputs_embeds=prefix_inputs_embeds,
                    teacher_tokens=teacher_tokens,
                    labels=labels,
                    input_tokens=input_tokens,
                    clear_mask=clear_mask,
                    clear_weight=current_clear_weight,
                )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, vl_gpt.parameters()), max_norm=1.0)
            optim.step()

            total_loss += float(loss.detach().cpu().item())
            
            # DEBUG: Log every step for first few steps
            if step <= 5:
                avg = total_loss / max(1, step)
                clr_ratio = total_clear / max(1, (total_clear + total_supervised))
                print(f"DEBUG Step {step} - loss: {float(loss.detach().cpu().item()):.4f}, running avg: {avg:.4f}")
                print(f"  Current batch: {supervised_count} supervised, {clear_count} clear")
                print(f"  Running totals: {total_supervised} supervised, {total_clear} clear, ratio: {clr_ratio:.2%}")
            elif step % 10 == 0:
                avg = total_loss / max(1, step)
                clr_ratio = total_clear / max(1, (total_clear + total_supervised))
                print(f"Step {step} - running loss: {avg:.4f}, CLEAR ratio: {clr_ratio:.2%}, supervised tokens so far: {total_supervised}, cw={current_clear_weight:.3f}")

            # Periodic checkpointing every 100 steps
            if step % 100 == 0:
                print(f"Saving checkpoint at epoch {epoch+1}, step {step}...")
                _save_checkpoint(epoch + 1, step)

            # Periodic validation every N steps (if enabled)
            if validate_every_steps and validate_every_steps > 0 and (step % validate_every_steps == 0):
                v_avg, v_actual, v_pred = _run_validation(epoch + 1, step)
                if auto_tune_clear_weight:
                    # Adjust clear weight to push predicted CLEAR toward actual CLEAR
                    delta = float(v_pred - v_actual)
                    old_cw = current_clear_weight
                    current_clear_weight = float(max(clear_weight_min, min(clear_weight_max, current_clear_weight - clear_weight_eta * delta)))
                    if abs(current_clear_weight - old_cw) > 1e-6:
                        print(f"Auto-tune clear_weight: {old_cw:.3f} -> {current_clear_weight:.3f} (delta={delta:+.3f}, eta={clear_weight_eta})")

        avg = total_loss / max(1, step)
        clr_ratio = total_clear / max(1, (total_clear + total_supervised))
        print(f"Epoch {epoch+1}/{epochs} - avg loss: {avg:.4f}, CLEAR ratio: {clr_ratio:.2%}, total supervised tokens: {total_supervised}")

        # Validation step (by epoch interval)
        if (epoch + 1) % max(1, int(validate_every)) == 0:
            v_avg, v_actual, v_pred = _run_validation(epoch + 1, None)
            if auto_tune_clear_weight:
                delta = float(v_pred - v_actual)
                old_cw = current_clear_weight
                current_clear_weight = float(max(clear_weight_min, min(clear_weight_max, current_clear_weight - clear_weight_eta * delta)))
                if abs(current_clear_weight - old_cw) > 1e-6:
                    print(f"Auto-tune clear_weight: {old_cw:.3f} -> {current_clear_weight:.3f} (delta={delta:+.3f}, eta={clear_weight_eta})")

    # Save fine-tuned weights
    vl_gpt.eval()
    # Save LoRA adapters
    vl_gpt.save_pretrained(output_dir)
    vl_chat_processor.tokenizer.save_pretrained(output_dir)
    # Save CLEAR head parameters and metadata if available
    meta = {}
    if hasattr(vl_gpt, "gen_head_clear"):
        gen_head_clear = vl_gpt.gen_head_clear
        torch.save(gen_head_clear.state_dict(), os.path.join(output_dir, "gen_head_clear.pt"))
        meta["has_clear_head"] = True
    with open(os.path.join(output_dir, "clear_meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf)
    print(f"Saved fine-tuned model to {output_dir}")


# Defaults for direct %run from a notebook
DEFAULT_JSONL = "/workspace/prepared_magicbrush.jsonl"
DEFAULT_MODEL = "FreedomIntelligence/Janus-4o-7B"
DEFAULT_OUTDIR = "/workspace/sft_janus4o_clear/"


if __name__ == "__main__":
    # Minimal default run (1 epoch, bs=1); customize in notebooks by importing train_sft
    train_sft(
        jsonl_path=DEFAULT_JSONL,
        model_path=DEFAULT_MODEL,
        output_dir=DEFAULT_OUTDIR,
        epochs=1,
        learning_rate=5e-5,
        batch_size=1,
        limit=-1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        clear_weight=0.3,
        validate_every_steps=100,
        auto_tune_clear_weight=True,
        clear_weight_eta=0.1, # adjust aggressiveness if needed
        clear_weight_min=0.1,
        clear_weight_max=1.0,
    )
