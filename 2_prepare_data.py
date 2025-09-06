import os
import json
from typing import List, Tuple

from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor



def select_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """
    Select a compute device and suitable dtype.
    - cuda: bfloat16
    - mps/cpu: float32
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def load_model_and_processor(model_path: str) -> Tuple[VLChatProcessor, MultiModalityCausalLM, torch.device, torch.dtype]:
    """
    Load the Janus-4o processor and model once.
    """
    device, dtype = select_device_and_dtype()

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

    # dtype selection: use bfloat16 on CUDA, float32 elsewhere
    load_kwargs = {"trust_remote_code": True}
    if device.type == "cuda":
        load_kwargs["torch_dtype"] = dtype

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs,
    )

    if device.type == "cuda":
        vl_gpt = vl_gpt.cuda().eval()
    else:
        vl_gpt = vl_gpt.to(device).eval()

    return vl_chat_processor, vl_gpt, device, dtype


def resize_image_rgb(image_path: str, size: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    return img.resize((size, size), Image.BICUBIC)


def resize_mask(mask_path: str, size: int) -> Image.Image:
    """
    Load mask if present and resize using nearest to avoid smoothing.
    Returns an RGB mask for potential visualization; not used for token logic.
    """
    img = Image.open(mask_path)
    # Preserve single-channel masks, but final mode doesn't impact token logic here
    if img.mode not in ("L", "1"):
        img = img.convert("L")
    return img.resize((size, size), Image.NEAREST)


def encode_image_to_tokens(
    image: Image.Image,
    vl_chat_processor: VLChatProcessor,
    vl_gpt: MultiModalityCausalLM,
    device: torch.device,
    dtype: torch.dtype,
) -> List[int]:
    """
    Encode a PIL image into 576 tokens using Janus vision tokenizer.
    """
    inputs = vl_chat_processor.image_processor([image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    if device.type == "cuda":
        pixel_values = pixel_values.to(dtype)

    quant, emb_loss, info = vl_gpt.gen_vision_model.encode(pixel_values)
    # info[2] is the flattened token sequence in prior experiments
    tokens: torch.Tensor = info[2].detach().reshape(pixel_values.shape[0], -1)
    tokens_list: List[int] = tokens.squeeze(0).to("cpu").tolist()
    return [int(t) for t in tokens_list]


def build_output_tokens(
    target_tokens: List[int],
    source_tokens: List[int],
) -> List[object]:
    """
    Build output tokens with "<CLEAR>" markers using token equality.
    """
    assert len(source_tokens) == len(target_tokens)
    out: List[object] = []
    for i, tok in enumerate(target_tokens):
        if int(tok) == int(source_tokens[i]):
            out.append("<CLEAR>")
        else:
            out.append(int(tok))
    return out


def list_sample_dirs(dataset_dir: str) -> List[str]:
    entries = [
        os.path.join(dataset_dir, d)
        for d in os.listdir(dataset_dir)
        if d.startswith("sample_") and os.path.isdir(os.path.join(dataset_dir, d))
    ]
    entries.sort()
    return entries


def prepare_dataset(
    dataset_dir: str,
    output_jsonl: str,
    model_path: str,
    image_size: int,
    patch_size: int,
    limit: int,
) -> None:
    vl_chat_processor, vl_gpt, device, dtype = load_model_and_processor(model_path)

    sample_dirs = list_sample_dirs(dataset_dir)
    if limit > 0:
        sample_dirs = sample_dirs[:limit]

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    written = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sample_path in sample_dirs:
            metadata_path = os.path.join(sample_path, "metadata.json")
            source_path = os.path.join(sample_path, "source.png")
            target_path = os.path.join(sample_path, "target.png")
            mask_path = os.path.join(sample_path, "mask.png")

            with open(metadata_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            instruction = str(meta.get("instruction", ""))

            source_img = resize_image_rgb(source_path, image_size)
            target_img = resize_image_rgb(target_path, image_size)
            if os.path.exists(mask_path):
                _ = resize_mask(mask_path, image_size)

            input_tokens = encode_image_to_tokens(
                source_img, vl_chat_processor, vl_gpt, device, dtype
            )
            target_tokens = encode_image_to_tokens(
                target_img, vl_chat_processor, vl_gpt, device, dtype
            )

            grid = image_size // patch_size
            assert len(input_tokens) == grid * grid, (
                f"Expected {grid*grid} tokens, got {len(input_tokens)} for {sample_path}"
            )
            assert len(target_tokens) == grid * grid, (
                f"Expected {grid*grid} tokens, got {len(target_tokens)} for {sample_path}"
            )

            output_tokens = build_output_tokens(
                target_tokens=target_tokens,
                source_tokens=input_tokens,
            )

            rec = {
                "instruction": instruction,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} examples to {output_jsonl}")


# Defaults for easy use in notebooks or via %run
DEFAULT_DATASET_DIR = "/workspace/magicbrush_examples/"
DEFAULT_OUT_JSONL = "/workspace/prepared_magicbrush.jsonl"
DEFAULT_MODEL_PATH = "FreedomIntelligence/Janus-4o-7B"
DEFAULT_IMAGE_SIZE = 384
DEFAULT_PATCH_SIZE = 16
DEFAULT_LIMIT = -1


if __name__ == "__main__":
    # Allows running via: %run 2_prepare_data.py
    prepare_dataset(
        dataset_dir=DEFAULT_DATASET_DIR,
        output_jsonl=DEFAULT_OUT_JSONL,
        model_path=DEFAULT_MODEL_PATH,
        image_size=DEFAULT_IMAGE_SIZE,
        patch_size=DEFAULT_PATCH_SIZE,
        limit=DEFAULT_LIMIT,
    )


