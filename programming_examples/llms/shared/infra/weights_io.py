# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Safetensors loading helpers shared by the llms/ weight loaders.

Ten `*_weights.py` files carried these two functions. `_load_tensor` was one
implementation in all ten; `_resolve_safetensor_files` looked like three, but
the variants differ only in whether a path pattern is bound to a temporary
first, so they are one implementation too.

`resolve_safetensor_files` tries the offline cache before the network, so a
cache hit does not print huggingface_hub's "Fetching N files" progress UI. The
`local_files_only=True` call returns whatever subset of `allow_patterns` is
already cached and does NOT raise when only some files match: a persistent CI
runner that previously ran `AutoConfig.from_pretrained` has `config.json`
cached but no safetensors, so the absence of `*.safetensors` is turned into
`LocalEntryNotFoundError` explicitly to force the network branch.
"""

import glob as glob_module
import os
from typing import List


def resolve_safetensor_files(model_path: str) -> List[str]:
    """Find all safetensors shard files for a model.

    Args:
        model_path: either a local directory, or a HuggingFace model id
            (e.g. "meta-llama/Llama-3.2-1B").

    Returns:
        Sorted list of absolute paths to .safetensors files.

    Raises:
        FileNotFoundError: if the directory (or the download) has none.
    """
    if os.path.isdir(model_path):
        files = sorted(glob_module.glob(os.path.join(model_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")
        return files

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    pattern_glob = "*.safetensors"
    try:
        local_dir = snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json"],
            local_files_only=True,
        )
        if not glob_module.glob(os.path.join(local_dir, pattern_glob)):
            raise LocalEntryNotFoundError(
                f"local cache for {model_path} has no .safetensors"
            )
    except LocalEntryNotFoundError:
        local_dir = snapshot_download(
            model_path, allow_patterns=["*.safetensors", "*.json"]
        )
    files = sorted(glob_module.glob(os.path.join(local_dir, pattern_glob)))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found after downloading {model_path}"
        )
    return files


def load_tensor(file_handle, key: str, dtype):
    """Load one tensor from an open safetensors file handle, cast to `dtype`.

    The safetensors library returns numpy arrays; a handle that returns a
    torch-like tensor is converted via its `.numpy()` first.
    """
    tensor = file_handle.get_tensor(key)
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return tensor.astype(dtype)
