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


class RBLNFishOmniCodecConfig(RBLNModelConfig):
    """
    Configuration class for the RBLN FishOmni codec decoder (modded DAC vocoder).

    The decoder compiles into two window graphs by default: `decoder` over the first `window_frames` code frames and
    `decoder_next` over each following window, which also takes the previous window's post_module K/V and its last
    `context_frames` post_module frames. With `compile_next=False` only `decoder` is compiled, for streams of one
    window. The host looks up the codebook vectors and feeds windows of `window_frames` frames; one frame becomes
    `samples_per_frame` (2048) samples.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        window_frames: int | None = None,
        context_frames: int | None = None,
        compile_next: bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int, optional): Batch size of the window graphs. Defaults to 1.
            window_frames (int, optional): Code frames per compiled window. Defaults to 128, the post_module
                attention window, so one window activation stays at 96 x 128 x 2048 values instead of the 3.0 GiB of
                8192 frames.
            context_frames (int, optional): Previous-window post_module frames the `decoder_next` graph recomputes
                the conv stack over. Defaults to 16; it must cover the conv receptive field (below 10 frames).
            compile_next (bool, optional): Whether to compile `decoder_next` (streams longer than one window).
                Defaults to True.
            kwargs: Additional arguments passed to the parent `RBLNModelConfig`.

        Raises:
            ValueError: If batch_size is not a positive integer or context_frames exceeds window_frames.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
        self.window_frames = window_frames or 128
        self.context_frames = context_frames or 16
        if self.context_frames > self.window_frames:
            raise ValueError("context_frames cannot exceed window_frames")
        self.compile_next = True if compile_next is None else compile_next


__all__ = ["RBLNFishOmniCodecConfig"]
