import json
import subprocess
from pathlib import Path


def ffprobe_video_stream(path: Path) -> dict[str, object]:
    result = subprocess.run(
        ["ffprobe", "-show_streams", "-print_format", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    video_streams = [stream for stream in data["streams"] if stream["codec_type"] == "video"]
    if not video_streams:
        raise RuntimeError("video stream not found")

    return video_streams[0]
