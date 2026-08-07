#!/usr/bin/env python3
"""
Create 2x2 preview videos for Insert USB real raw episodes.

Recommended usage from the OpenPI repo root:

    cd /shareNFS_40/wyf/workplace/openpi

    uv run examples/real_preprocess/make_preview_videos.py \
        --raw-dir /shareNFS_40/wyf/data/real/insert_usb_gelsight \
        --overwrite

By default this writes one video inside each episode directory:

    /shareNFS_40/wyf/data/real/insert_usb_gelsight/000000/preview_2x2.mp4

Each video frame is:

    main | wrist
    left | right

where every image is resized to 224x224 before tiling.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_RAW_DIR = Path("/shareNFS_40/wyf/data/real/insert_usb_gelsight")
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_VIDEO_NAME = "preview_2x2.mp4"
DEFAULT_FPS = 15.0
DEFAULT_ENCODER = "h264"

IMAGE_STREAMS = (
    ("main", Path("main/color")),
    ("wrist", Path("wrist/color")),
    ("left", Path("tactile/left/rectify")),
    ("right", Path("tactile/right/rectify")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 2x2 preview videos for Insert USB real raw episodes.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing raw episode folders. Default: {DEFAULT_RAW_DIR}",
    )
    parser.add_argument(
        "--episode-ids",
        nargs="*",
        default=None,
        help="Optional episode ids to process, e.g. 000000 000004.",
    )
    parser.add_argument(
        "--episode-start",
        type=int,
        default=None,
        help="First numeric episode id to process, inclusive. Ignored when --episode-ids is set.",
    )
    parser.add_argument(
        "--episode-end",
        type=int,
        default=None,
        help="Last numeric episode id to process, inclusive. Ignored when --episode-ids is set.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Process only the first N sorted episodes after filtering.",
    )
    parser.add_argument(
        "--video-name",
        default=DEFAULT_VIDEO_NAME,
        help=f"Output video filename inside each episode directory. Default: {DEFAULT_VIDEO_NAME}",
    )
    parser.add_argument(
        "--encoder",
        choices=("h264", "mp4v", "mjpg", "xvid"),
        default=DEFAULT_ENCODER,
        help=(
            "Video encoder. h264 uses imageio-ffmpeg/libx264 and is easiest to preview in VS Code. "
            "mp4v/mjpg/xvid use OpenCV VideoWriter. Default: h264."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Output preview video FPS. Default: {DEFAULT_FPS}.",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=DEFAULT_IMAGE_SIZE,
        help=f"Resize each stream before tiling. Default: {DEFAULT_IMAGE_SIZE[0]} {DEFAULT_IMAGE_SIZE[1]}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing preview videos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned videos without writing files.",
    )
    return parser.parse_args()


def progress(items: Iterable, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def sorted_episode_dirs(raw_dir: Path, episode_ids: list[str] | None) -> list[Path]:
    if episode_ids is not None:
        episodes = [raw_dir / episode_id for episode_id in episode_ids]
    else:
        episodes = sorted((path for path in raw_dir.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda p: int(p.name))

    missing = [str(path) for path in episodes if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing episode directories: {missing}")
    if not episodes:
        raise FileNotFoundError(f"No episode directories found in {raw_dir}")
    return episodes


def filter_episode_range(episodes: list[Path], episode_start: int | None, episode_end: int | None) -> list[Path]:
    if episode_start is None and episode_end is None:
        return episodes

    start = 0 if episode_start is None else episode_start
    end = 10**12 if episode_end is None else episode_end
    if start < 0 or end < 0:
        raise ValueError(f"Episode range must be non-negative, got {start}-{end}")
    if start > end:
        raise ValueError(f"--episode-start must be <= --episode-end, got {start}>{end}")

    filtered = [path for path in episodes if start <= int(path.name) <= end]
    if not filtered:
        raise FileNotFoundError(f"No episode directories found in requested range {start}-{end}")
    return filtered


def sorted_jpgs(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    paths = sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if not paths:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}")
    return paths


def load_stream_paths(ep_dir: Path) -> dict[str, list[Path]]:
    streams = {name: sorted_jpgs(ep_dir / rel_dir) for name, rel_dir in IMAGE_STREAMS}
    lengths = {name: len(paths) for name, paths in streams.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{ep_dir}: stream frame counts do not match: {lengths}")

    num_frames = next(iter(lengths.values()))
    for name, paths in streams.items():
        for idx, path in enumerate(paths):
            expected = f"{idx:06d}"
            if path.stem != expected:
                raise ValueError(f"{ep_dir}/{name}: expected {expected}.jpg at index {idx}, got {path.name}")

    return streams


def read_resized_bgr(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV failed to read image: {path}")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {image_size}")
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def tile_frame(main: np.ndarray, wrist: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    top = np.concatenate([main, wrist], axis=1)
    bottom = np.concatenate([left, right], axis=1)
    return np.concatenate([top, bottom], axis=0)


def iter_tiled_frames(streams: dict[str, list[Path]], image_size: tuple[int, int]):
    num_frames = len(streams["main"])
    for idx in range(num_frames):
        yield tile_frame(
            read_resized_bgr(streams["main"][idx], image_size),
            read_resized_bgr(streams["wrist"][idx], image_size),
            read_resized_bgr(streams["left"][idx], image_size),
            read_resized_bgr(streams["right"][idx], image_size),
        )


def write_h264_video(output_path: Path, streams: dict[str, list[Path]], image_size: tuple[int, int], fps: float) -> None:
    width, height = image_size
    output_size = (width * 2, height * 2)
    tmp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{output_size[0]}x{output_size[1]}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in iter_tiled_frames(streams, image_size):
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
        returncode = process.wait()
    except Exception:
        process.kill()
        process.wait()
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    if returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"ffmpeg failed while writing {output_path}:\n{stderr}")

    tmp_path.replace(output_path)


def write_opencv_video(
    output_path: Path,
    streams: dict[str, list[Path]],
    image_size: tuple[int, int],
    fps: float,
    encoder: str,
) -> None:
    width, height = image_size
    output_size = (width * 2, height * 2)
    fourcc_by_encoder = {
        "mp4v": "mp4v",
        "mjpg": "MJPG",
        "xvid": "XVID",
    }
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*fourcc_by_encoder[encoder]), fps, output_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for frame in iter_tiled_frames(streams, image_size):
            writer.write(frame)
    finally:
        writer.release()


def make_episode_video(
    ep_dir: Path,
    video_name: str,
    image_size: tuple[int, int],
    fps: float,
    encoder: str,
    overwrite: bool,
    dry_run: bool,
) -> int:
    streams = load_stream_paths(ep_dir)
    num_frames = len(streams["main"])
    output_path = ep_dir / video_name

    if output_path.exists() and not overwrite:
        print(f"skip existing: {output_path}")
        return 0

    if dry_run:
        print(f"would write: {output_path} ({num_frames} frames, {fps:g} fps, encoder={encoder})")
        return num_frames

    if encoder == "h264":
        write_h264_video(output_path, streams, image_size, fps)
    else:
        write_opencv_video(output_path, streams, image_size, fps, encoder)

    print(f"wrote: {output_path} ({num_frames} frames, {fps:g} fps, encoder={encoder})")
    return num_frames


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    image_size = tuple(args.image_size)

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")

    episodes = sorted_episode_dirs(raw_dir, args.episode_ids)
    if args.episode_ids is None:
        episodes = filter_episode_range(episodes, args.episode_start, args.episode_end)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    print(f"Raw dir: {raw_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Video name: {args.video_name}")
    print(f"Encoder: {args.encoder}")
    print(f"FPS: {args.fps:g}")
    print(f"Tile size: {image_size[0] * 2}x{image_size[1] * 2}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Dry run: {args.dry_run}")

    total_frames = 0
    for ep_dir in progress(episodes, desc="Creating videos"):
        total_frames += make_episode_video(
            ep_dir=ep_dir,
            video_name=args.video_name,
            image_size=image_size,
            fps=args.fps,
            encoder=args.encoder,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    print(f"Done. Processed {len(episodes)} episodes, wrote/planned {total_frames} frames.")


if __name__ == "__main__":
    main()
