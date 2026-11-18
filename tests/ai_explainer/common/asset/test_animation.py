from dataclasses import dataclass

import pytest
import torch
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection

import testing
import testing.example as aix
from testing.example.asset import Animation, Outputs, Viewport


@pytest.mark.video
class TestAnimation:
    def test_animation(self, outputs: Outputs, goldens: Outputs) -> None:
        fig, ax = aix.asset.figure(figsize=(1, 1))
        aix.asset.frame_empty(ax, viewport=Viewport(xlim=(0, 3), ylim=(0, 3)))
        dot = aix.asset.plot_point(ax, torch.tensor(1.0), torch.tensor(1.0))
        animator = ExampleAnimator(dot)
        anim = Animation(fig, animator, init_frames=2, active_frames=4, fps=3)

        testing.assert_matches_golden(outputs, goldens, anim, "animation")

    def test_save(self, outputs: Outputs) -> None:
        fig, _ = aix.asset.figure(figsize=(0.5, 1))
        anim = Animation(fig, NoOpAnimator(), init_frames=2, active_frames=3, fps=3)
        anim.save(outputs.mp4("animation"))

        stream = testing.ffprobe_video_stream(outputs.mp4("animation"))
        assert stream["width"] == 150
        assert stream["height"] == 300
        assert stream["nb_frames"] == "5"
        assert stream["avg_frame_rate"] == "3/1"
        assert stream["codec_name"] == "h264"
        assert stream["pix_fmt"] == "yuv420p"


@dataclass(frozen=True)
class ExampleAnimator:
    dot: PathCollection

    def animate(self, frame: int) -> list[Artist]:
        pos = 1 + 0.25 * frame
        self.dot.set_offsets([pos, pos])
        return [self.dot]


class NoOpAnimator:
    def animate(self, _frame: int) -> list[Artist]:
        return []
