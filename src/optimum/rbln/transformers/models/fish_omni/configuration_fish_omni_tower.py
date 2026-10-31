# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from ....configuration_utils import RBLNModelConfig


class RBLNFishOmniTowerConfig(RBLNModelConfig):
    """
    Configuration class for the RBLN FishOmni codebook tower (codebook-MTP audio decoder).

    The tower compiles into a teacher-forced `prefill` graph over every layer and one decode graph per residual codebook
    (`decode_1` = shared layers + residual layer 1 + head 1, `decode_k` = residual layer k + head k). All graphs share
    one static KV cache per layer.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        max_seq_len: int | None = None,
        prefill_chunk_size: int | None = None,
        decode_graphs: list[int] | None = None,
        compile_prefill: bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int, optional): Decode batch size. Defaults to 1.
            max_seq_len (int, optional): KV cache length per sequence (frames). Defaults to 8192.
            prefill_chunk_size (int, optional): Rows per prefill call. Defaults to 128.
            decode_graphs (list[int], optional): Residual indices k to compile decode graphs for (1-based). Defaults
                to every residual codebook. The text-to-speech loop needs all of them.
            compile_prefill (bool, optional): Whether to compile the prefill graph. Defaults to True.
            kwargs: Additional arguments passed to the parent `RBLNModelConfig`.

        Raises:
            ValueError: If batch_size is not a positive integer or max_seq_len is not a multiple of
                prefill_chunk_size.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
        self.max_seq_len = max_seq_len or 8192
        self.prefill_chunk_size = prefill_chunk_size or 128
        if self.max_seq_len % self.prefill_chunk_size:
            raise ValueError("max_seq_len must be a multiple of prefill_chunk_size")
        self.decode_graphs = list(decode_graphs) if decode_graphs is not None else None
        self.compile_prefill = True if compile_prefill is None else compile_prefill


__all__ = ["RBLNFishOmniTowerConfig"]
