# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from vllm.utils.system_utils import _maybe_force_spawn, unique_filepath


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


def test_maybe_force_spawn_with_wsl(monkeypatch):
    """Test that _maybe_force_spawn() forces spawn when WSL is detected."""

    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)

    # Mock in_wsl to return True, and ensure other conditions don't apply
    with patch("vllm.utils.system_utils.in_wsl", return_value=True), \
         patch("vllm.utils.system_utils.is_in_ray_actor", return_value=False), \
         patch("vllm.utils.system_utils.cuda_is_initialized", return_value=False), \
         patch("vllm.utils.system_utils.xpu_is_initialized", return_value=False):
        _maybe_force_spawn()

    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
