from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Protocol

from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.figure import Figure


@dataclass(frozen=True)
class Animation:
    """Animated figure.

    Frame 0 is the original figure, which is held for init_frames.
    The animator is used for active frames, starting at frame 1.
    """

    fig: Figure
    animator: Animator
    _: KW_ONLY
    init_frames: int
    active_frames: int
    fps: int

    def save(self, path: Path) -> None:
        """Save the animation as an MP4 file."""
        frames = self.init_frames + self.active_frames
        anim = FuncAnimation(self.fig, self._animate, frames)
        anim.save(
            path,
            dpi=300,
            fps=self.fps,
            writer="ffmpeg",
            codec="libx264",
            extra_args=[
                *["-pix_fmt", "yuv420p"],
                *["-movflags", "+faststart"],
                *["-x264-params", "asm=0"],  # for reproducibility
                *["-threads", "6"],  # for reproducibility
            ],
        )

    def _animate(self, frame: int) -> list[Artist]:
        frame -= self.init_frames - 1
        return self.animator.animate(frame) if frame > 0 else []


class Animator(Protocol):
    def animate(self, frame: int, /) -> list[Artist]: ...
