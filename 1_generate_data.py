import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Image as HFImage
from datasets import Dataset, load_dataset
from PIL import Image
import shutil

os.environ["HF_HOME"] = "/workspace"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hub"
os.environ["HF_HUB_CACHE"] = "/workspace/hub"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)



"""
Jupyter-friendly utilities to load and filter osunlp/MagicBrush and produce
examples with PIL images for source/mask/target.

Basic usage in a notebook:

from 1_generate_data import load_magicbrush_examples
examples, info = load_magicbrush_examples(
    split="train",
    whitelist=["add", "insert"],
    blacklist=["remove"],
    whitelist_mode="any",
    blacklist_mode="any",
    shuffle=True,
    seed=42,
    limit=10,
    save_dir=None,
)

Each element in examples is a dict with keys: instruction, source, target, mask.
info includes detected column names.
"""


def split_to_keywords(value: str) -> List[str]:
    if not value:
        return []
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def load_magicbrush(split: str) -> Dataset:
    # Rely on datasets' native slicing (e.g., "train[:1000]") to avoid
    # reconstructing from streaming, which can force PIL re-encoding.
    return load_dataset("osunlp/MagicBrush", split=split)


def detect_columns(ds: Dataset) -> Tuple[str, Optional[str], str]:
    """
    Detect instruction, mask, and target/source columns.

    Returns a tuple: (instruction_col, mask_col_or_none, target_col)

    We will also return source via a separate helper because the dataset names
    may vary between `source`, `source_img`, `target`, `target_img`, `mask`, `mask_img`.
    """
    features = ds.features
    columns = list(features.keys())

    # Instruction
    instruction_candidates = [
        "instruction",
        "prompt",
        "edit_instruction",
    ]
    instruction_col = next((c for c in instruction_candidates if c in features), None)
    if instruction_col is None:
        # fallback: first string column
        for c in columns:
            if str(features[c].dtype) == "string":
                instruction_col = c
                break
    if instruction_col is None:
        raise ValueError("Could not detect instruction column.")

    # Image columns (HF Image feature)
    image_cols = [c for c in columns if isinstance(features[c], HFImage)]

    # Map common names
    name_map = {c.lower(): c for c in image_cols}
    source_col = None
    for candidate in ["source", "source_img", "src", "image", "input", "input_img"]:
        if candidate in name_map:
            source_col = name_map[candidate]
            break
    if source_col is None and image_cols:
        source_col = image_cols[0]

    mask_col = None
    for candidate in ["mask", "mask_img", "mask_image"]:
        if candidate in name_map:
            mask_col = name_map[candidate]
            break

    target_col = None
    for candidate in ["target_img", "target", "edited", "output", "output_img"]:
        if candidate in name_map:
            target_col = name_map[candidate]
            break
    if target_col is None and len(image_cols) >= 2:
        target_col = image_cols[-1]

    if source_col is None or target_col is None:
        raise ValueError(
            f"Could not detect image columns. Found image columns: {image_cols}"
        )

    return instruction_col, mask_col, target_col


def cast_images(ds: Dataset) -> Dataset:
    """
    Ensure all Image feature columns are decoded to PIL on access.
    """
    for c, feat in ds.features.items():
        if isinstance(feat, HFImage):
            ds = ds.cast_column(c, HFImage(decode=True))
    return ds


def get_source_column(ds: Dataset) -> str:
    features = ds.features
    image_cols = [c for c in features if isinstance(features[c], HFImage)]
    name_map = {c.lower(): c for c in image_cols}
    for candidate in ["source", "source_img", "src", "image", "input", "input_img"]:
        if candidate in name_map:
            return name_map[candidate]
    if image_cols:
        return image_cols[0]
    raise ValueError("No image columns found for source.")


def instruction_matches(
    text: str,
    whitelist: List[str],
    blacklist: List[str],
    whitelist_mode: str,
    blacklist_mode: str,
) -> bool:
    t = (text or "").lower()

    # Whitelist: if provided, require any/all per mode. If empty, pass.
    if whitelist:
        if whitelist_mode == "any":
            if not any(k in t for k in whitelist):
                return False
        else:  # all
            if not all(k in t for k in whitelist):
                return False

    # Blacklist: if provided, exclude if any/all present per mode.
    if blacklist:
        if blacklist_mode == "any":
            if any(k in t for k in blacklist):
                return False
        else:  # all
            if all(k in t for k in blacklist):
                return False

    return True


def filter_dataset(
    ds: Dataset,
    instruction_col: str,
    whitelist: List[str],
    blacklist: List[str],
    whitelist_mode: str,
    blacklist_mode: str,
    turn_index: Optional[int] = None,
) -> Dataset:
    def _keep(batch: Dict[str, List]) -> List[bool]:
        texts = batch[instruction_col]
        result = [
            instruction_matches(t, whitelist, blacklist, whitelist_mode, blacklist_mode)
            for t in texts
        ]
        
        # Apply turn_index filter if specified
        if turn_index is not None and "turn_index" in batch:
            turn_indices = batch["turn_index"]
            result = [
                result[i] and (turn_indices[i] == turn_index)
                for i in range(len(result))
            ]
        
        return result

    return ds.filter(_keep, batched=True)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_image_any(img_or_ref: Any, out_path: str) -> None:
    """
    Save an image reference to disk efficiently.
    - If it's a PIL.Image with a valid filename on disk, copy the file.
    - If it's a PIL.Image without a filename, save as fast-compression PNG.
    - If it's a dict with 'path' or 'bytes' (HF Image streaming), copy path or write bytes.
    - If it's a string path, copy the file.
    """
    # PIL.Image
    if isinstance(img_or_ref, Image.Image):
        if getattr(img_or_ref, "filename", None) and os.path.exists(img_or_ref.filename):
            shutil.copy2(img_or_ref.filename, out_path)
        else:
            img_or_ref.save(out_path, compress_level=1)
        return

    # HF Image dict
    if isinstance(img_or_ref, dict):
        path_value = img_or_ref.get("path")
        bytes_value = img_or_ref.get("bytes")
        if path_value and os.path.exists(path_value):
            shutil.copy2(path_value, out_path)
            return
        if bytes_value is not None:
            with open(out_path, "wb") as f:
                f.write(bytes_value)
            return
        raise ValueError("Unsupported image dict format: expected 'path' or 'bytes'.")

    # String path
    if isinstance(img_or_ref, str):
        if os.path.exists(img_or_ref):
            shutil.copy2(img_or_ref, out_path)
            return
        raise ValueError(f"Image path does not exist: {img_or_ref}")

    raise ValueError(f"Unsupported image reference type: {type(img_or_ref)}")


def save_example(
    base_dir: str,
    idx: int,
    instruction: str,
    source: Any,
    target: Any,
    mask: Optional[Any],
    extra: Optional[Dict] = None,
) -> None:
    sample_dir = os.path.join(base_dir, f"sample_{idx:06d}")
    os.makedirs(sample_dir, exist_ok=True)
    source_out = os.path.join(sample_dir, "source.png")
    save_image_any(source, source_out)
    if mask is not None:
        mask_out = os.path.join(sample_dir, "mask.png")
        save_image_any(mask, mask_out)
    target_out = os.path.join(sample_dir, "target.png")
    save_image_any(target, target_out)

    meta = {"instruction": instruction}
    if extra:
        meta.update(extra)
    with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_magicbrush_examples(
    split: str = "train",
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None,
    whitelist_mode: str = "any",
    blacklist_mode: str = "any",
    turn_index: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    limit: Optional[int] = 10,
    save_dir: Optional[str] = None,
    streaming_limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    """
    Load MagicBrush, filter by instruction keywords, and return examples with PIL images.

    Returns (examples, info). Each example is a dict with keys:
      - instruction: str
      - source: PIL.Image
      - target: PIL.Image
      - mask: Optional[PIL.Image]

    info contains detected column names: instruction, source, mask, target.
    """
    wl = [w.lower() for w in (whitelist or [])]
    bl = [b.lower() for b in (blacklist or [])]

    # Streaming mode: fetch only first N without downloading the full split
    if streaming_limit is not None and streaming_limit > 0:
        from datasets import IterableDataset

        # Parse split range like "train[:500]" -> base_split, start, end
        def _parse_split_range(s: str) -> Tuple[str, int, Optional[int]]:
            if "[" in s and ":" in s:
                base = s.split("[")[0]
                slice_part = s.split("[")[1].rstrip("]")
                parts = slice_part.split(":")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if len(parts) > 1 and parts[1] else None
                return base, start, end
            return s, 0, None

        base_split, start_idx, end_idx = _parse_split_range(split)
        ds_stream = load_dataset("osunlp/MagicBrush", split=base_split, streaming=True)

        # Prime one element to detect columns
        it = iter(ds_stream)
        try:
            first_item = next(it)
        except StopIteration:
            return [], {"instruction": None, "source": None, "mask": None, "target": None, "total_after_filter": 0, "returned": 0}

        # Helper to detect columns from a single example
        def _detect_cols_from_item(item: Dict[str, Any]) -> Tuple[str, Optional[str], str, str]:
            keys = list(item.keys())
            # instruction
            instruction_candidates = ["instruction", "prompt", "edit_instruction"]
            instr_col_local = next((c for c in instruction_candidates if c in item), None)
            if instr_col_local is None:
                # fallback: first string-like
                instr_col_local = next((k for k in keys if isinstance(item[k], str)), keys[0])

            # image-like detection
            def is_image_like(v: Any) -> bool:
                if isinstance(v, Image.Image):
                    return True
                if isinstance(v, dict) and ("path" in v or "bytes" in v):
                    return True
                return False

            image_keys = [k for k in keys if is_image_like(item[k])]
            name_map = {k.lower(): k for k in image_keys}

            src_col_local = None
            for candidate in ["source", "source_img", "src", "image", "input", "input_img"]:
                if candidate in name_map:
                    src_col_local = name_map[candidate]
                    break
            if src_col_local is None and image_keys:
                src_col_local = image_keys[0]

            mask_col_local = None
            for candidate in ["mask", "mask_img", "mask_image"]:
                if candidate in name_map:
                    mask_col_local = name_map[candidate]
                    break

            tgt_col_local = None
            for candidate in ["target_img", "target", "edited", "output", "output_img"]:
                if candidate in name_map:
                    tgt_col_local = name_map[candidate]
                    break
            if tgt_col_local is None and len(image_keys) >= 2:
                tgt_col_local = image_keys[-1]

            if src_col_local is None or tgt_col_local is None:
                raise ValueError("Could not detect image columns in streaming mode.")

            return instr_col_local, mask_col_local, tgt_col_local, src_col_local

        instruction_col, maybe_mask_col, target_col, source_col = _detect_cols_from_item(first_item)

        # Rebuild an iterator that yields the first item back, then the rest
        def _chain_first_and_rest(first: Dict[str, Any], rest_iter):
            yield first
            for x in rest_iter:
                yield x

        combined_iter = _chain_first_and_rest(first_item, it)

        # Handle start/end slicing by manual skipping/limit
        current_index = 0
        matched = 0
        max_needed = streaming_limit
        if end_idx is not None:
            range_count = max(0, end_idx - start_idx)
            max_needed = min(max_needed, range_count)

        # Prepare save context if needed
        if save_dir:
            ensure_dir(save_dir)
            jsonl_path = os.path.join(save_dir, "metadata.jsonl")
            jsonl_file = open(jsonl_path, "w", encoding="utf-8")
        else:
            jsonl_file = None

        examples: List[Dict[str, Any]] = []
        try:
            for item in combined_iter:
                # apply start offset
                if current_index < start_idx:
                    current_index += 1
                    continue

                instr_val = item.get(instruction_col)

                # turn_index filtering if present
                if turn_index is not None and ("turn_index" in item) and (item["turn_index"] != turn_index):
                    current_index += 1
                    continue

                # whitelist/blacklist filtering
                if not instruction_matches(str(instr_val or ""), wl, bl, whitelist_mode, blacklist_mode):
                    current_index += 1
                    continue

                src_val = item.get(source_col)
                msk_val = item.get(maybe_mask_col) if maybe_mask_col and (maybe_mask_col in item) else None
                tgt_val = item.get(target_col)

                # Persist if requested
                if jsonl_file is not None:
                    save_example(
                        base_dir=save_dir,  # type: ignore[arg-type]
                        idx=matched,
                        instruction=str(instr_val),
                        source=src_val,
                        target=tgt_val,
                        mask=msk_val,
                    )
                    rec = {"index": matched, "instruction": str(instr_val)}
                    jsonl_file.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Return in-memory refs (do not force decode)
                examples.append(
                    {
                        "instruction": str(instr_val),
                        "source": src_val,
                        "mask": msk_val,
                        "target": tgt_val,
                    }
                )

                matched += 1
                current_index += 1
                if matched >= max_needed:
                    break

                # Stop early if end bound reached
                if end_idx is not None and current_index >= end_idx:
                    break
        finally:
            if jsonl_file is not None:
                jsonl_file.close()

        info = {
            "instruction": instruction_col,
            "source": source_col,
            "mask": maybe_mask_col,
            "target": target_col,
            "total_after_filter": matched,
            "returned": len(examples),
        }
        return examples, info

    # Non-streaming mode: load full split metadata then slice in-memory
    ds = load_magicbrush(split)
    if shuffle:
        ds = ds.shuffle(seed=seed)

    ds = cast_images(ds)
    instruction_col, maybe_mask_col, target_col = detect_columns(ds)
    source_col = get_source_column(ds)

    if wl or bl or turn_index is not None:
        ds = filter_dataset(
            ds,
            instruction_col=instruction_col,
            whitelist=wl,
            blacklist=bl,
            whitelist_mode=whitelist_mode,
            blacklist_mode=blacklist_mode,
            turn_index=turn_index,
        )

    total = len(ds)
    n = total if (limit is None or limit <= 0) else min(limit, total)

    if save_dir:
        ensure_dir(save_dir)
        jsonl_path = os.path.join(save_dir, "metadata.jsonl")
        jsonl_file_ctx = open(jsonl_path, "w", encoding="utf-8")
    else:
        jsonl_file_ctx = None

    def to_pil(x):
        if isinstance(x, Image.Image):
            return x
        if isinstance(x, dict) and "bytes" in x:
            from io import BytesIO

            return Image.open(BytesIO(x["bytes"]))
        return x

    examples: List[Dict[str, Any]] = []
    if jsonl_file_ctx is not None:
        with jsonl_file_ctx as jsonl_file:
            for i in range(n):
                example = ds[i]
                instr = example[instruction_col]
                src_img = to_pil(example[source_col])
                msk_img = to_pil(example[maybe_mask_col]) if maybe_mask_col and example.get(maybe_mask_col) is not None else None
                tgt_img = to_pil(example[target_col])

                save_example(
                    base_dir=save_dir,  # type: ignore[arg-type]
                    idx=i,
                    instruction=str(instr),
                    source=src_img,
                    target=tgt_img,
                    mask=msk_img,
                )
                rec = {"index": i, "instruction": str(instr)}
                jsonl_file.write(json.dumps(rec, ensure_ascii=False) + "\n")

                examples.append(
                    {
                        "instruction": str(instr),
                        "source": src_img,
                        "mask": msk_img,
                        "target": tgt_img,
                    }
                )
    else:
        for i in range(n):
            example = ds[i]
            instr = example[instruction_col]
            src_img = to_pil(example[source_col])
            msk_img = to_pil(example[maybe_mask_col]) if maybe_mask_col and example.get(maybe_mask_col) is not None else None
            tgt_img = to_pil(example[target_col])
            examples.append(
                {
                    "instruction": str(instr),
                    "source": src_img,
                    "mask": msk_img,
                    "target": tgt_img,
                }
            )

    info = {
        "instruction": instruction_col,
        "source": source_col,
        "mask": maybe_mask_col,
        "target": target_col,
        "total_after_filter": total,
        "returned": len(examples),
    }
    return examples, info


def compute_mask_coverage_percent(mask_img: Optional[Image.Image]) -> Optional[float]:
    """
    Compute percentage of pixels being edited in a mask image.
    Assumes mask is either binary or grayscale where non-zero means edited.
    Returns None if mask is None.
    """
    if mask_img is None:
        return None
    arr = np.array(mask_img)
    if arr.ndim == 3:
        # If RGB(A), convert to single channel by max across channels
        arr = arr.max(axis=-1)
    total = arr.size
    edited = (arr > 0).sum()
    return float(edited) / float(total) * 100.0


def mask_coverage_distribution(examples: List[Dict[str, Any]]) -> List[float]:
    """
    Compute list of mask coverage percentages for provided examples.
    Examples with None mask are skipped.
    """
    out: List[float] = []
    for ex in examples:
        pct = compute_mask_coverage_percent(ex.get("mask"))
        if pct is not None:
            out.append(pct)
    return out


def plot_mask_coverage_hist(coverages: List[float], bins: int = 20) -> None:
    """
    Plot a histogram of mask coverage percentages using matplotlib.
    """
    if not coverages:
        print("No mask coverages to plot (no masks present or all None).")
        return
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist(coverages, bins=bins, color="#4C78A8", edgecolor="white")
    plt.xlabel("Mask coverage (% of pixels edited)")
    plt.ylabel("Count")
    plt.title("MagicBrush mask coverage distribution")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def view_random_example(save_dir: str, sample_idx: Optional[int] = None) -> None:
    """
    Load and display a random example from the saved dataset directory.
    Shows source, mask (if available), target images and instruction.
    
    Args:
        save_dir: Directory containing saved examples
        sample_idx: Specific sample index to view, or None for random
    """
    import matplotlib.pyplot as plt
    import random
    
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} does not exist.")
        return
    
    # Find all sample directories
    sample_dirs = [d for d in os.listdir(save_dir) if d.startswith("sample_") and os.path.isdir(os.path.join(save_dir, d))]
    
    if not sample_dirs:
        print(f"No sample directories found in {save_dir}")
        return
    
    # Select sample
    if sample_idx is not None:
        target_dir = f"sample_{sample_idx:06d}"
        if target_dir not in sample_dirs:
            print(f"Sample {sample_idx} not found. Available samples: 0-{len(sample_dirs)-1}")
            return
        sample_dir = target_dir
    else:
        sample_dir = random.choice(sample_dirs)
    
    sample_path = os.path.join(save_dir, sample_dir)
    
    # Load metadata
    metadata_path = os.path.join(sample_path, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"No metadata.json found in {sample_path}")
        return
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    instruction = metadata.get("instruction", "No instruction")
    
    # Load images
    source_path = os.path.join(sample_path, "source.png")
    target_path = os.path.join(sample_path, "target.png")
    mask_path = os.path.join(sample_path, "mask.png")
    
    if not os.path.exists(source_path) or not os.path.exists(target_path):
        print(f"Missing required images in {sample_path}")
        return
    
    source_img = Image.open(source_path)
    target_img = Image.open(target_path)
    has_mask = os.path.exists(mask_path)
    mask_img = Image.open(mask_path) if has_mask else None
    
    # Create subplot layout
    num_cols = 3 if has_mask else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    if num_cols == 1:
        axes = [axes]
    
    # Display source
    axes[0].imshow(source_img)
    axes[0].set_title("Source (Before)")
    axes[0].axis("off")
    
    # Display mask if available
    if has_mask:
        axes[1].imshow(mask_img, cmap="gray" if mask_img.mode in ["L", "1"] else None)
        axes[1].set_title("Mask")
        axes[1].axis("off")
        target_idx = 2
    else:
        target_idx = 1
    
    # Display target
    axes[target_idx].imshow(target_img)
    axes[target_idx].set_title("Target (After)")
    axes[target_idx].axis("off")
    
    # Add instruction as title
    fig.suptitle(f"Sample {sample_dir}\nInstruction: {instruction}", fontsize=12, wrap=True)
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed sample: {sample_dir}")
    print(f"Instruction: {instruction}")



# Sample usage
# examples, info = load_magicbrush_examples(
#     split="train",                 # base split only; slicing is handled in-stream
#     whitelist=[],
#     blacklist=["word"],
#     whitelist_mode="any",
#     blacklist_mode="any",
#     turn_index=1,
#     shuffle=False,                 # shuffle not supported efficiently in streaming
#     seed=42,
#     limit=None,                    # not used in streaming mode
#     save_dir='/workspace/magicbrush_examples/',
#     streaming_limit=500,           # fetch only first 500 that pass filters
# )