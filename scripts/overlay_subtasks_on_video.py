#!/usr/bin/env python3
"""Burn the active low-level subtask into a RoboTwin rollout video."""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from fractions import Fraction
from pathlib import Path


def _probe_video(video_path: Path) -> tuple[float, int, int, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    frame_count = int(stream.get("nb_frames") or 0)
    duration = float(stream.get("duration") or 0.0)
    if duration <= 0 and frame_count > 0:
        duration = frame_count / fps
    return fps, int(stream["width"]), int(stream["height"]), duration


def _read_transitions(
    trace_path: Path,
    *,
    episode_index: int,
) -> list[tuple[int, str]]:
    transitions: list[tuple[int, str]] = []
    previous = None
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("episode_index", -1)) != int(episode_index):
                continue
            instruction = str(
                row.get("high_level_current_instruction")
                or row.get("low_level_instruction")
                or ""
            ).strip()
            if not instruction or instruction == previous:
                continue
            transitions.append((int(row["frame_id"]), instruction))
            previous = instruction
    if not transitions:
        raise ValueError(f"No subtask instructions found in {trace_path}")
    return transitions


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _ass_text(instruction: str, wrap_width: int) -> str:
    escaped = (
        instruction.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", " ")
    )
    lines = textwrap.wrap(escaped, width=wrap_width, break_long_words=False)
    body = r"\N".join(lines)
    return r"{\b1\c&H6DE8FF&}CURRENT SUBTASK{\r}\N" + body


def _write_ass(
    ass_path: Path,
    transitions: list[tuple[int, str]],
    *,
    fps: float,
    width: int,
    height: int,
    duration: float,
) -> None:
    # RoboTwin writes one initial video frame before action frame 0.
    starts = [0.0]
    starts.extend((frame_id + 1) / fps for frame_id, _ in transitions[1:])
    font_size = max(18, round(height * 0.047))
    wrap_width = max(42, round(width / (font_size * 0.57)))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Subtask,DejaVu Sans,{font_size},&H00FFFFFF,&H00FFFFFF,&H50000000,&HA8000000,0,0,0,0,100,100,0,0,3,1,0,8,24,24,18,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for index, (_, instruction) in enumerate(transitions):
        start = starts[index]
        end = starts[index + 1] if index + 1 < len(starts) else duration
        events.append(
            "Dialogue: 0,"
            f"{_ass_time(start)},{_ass_time(end)},Subtask,,0,0,0,,"
            f"{_ass_text(instruction, wrap_width)}"
        )
    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()

    video = args.video.resolve()
    trace = args.trace.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fps, source_width, source_height, duration = _probe_video(video)
    output_width = args.width if args.width > 0 else source_width
    output_height = round(source_height * output_width / source_width / 2) * 2
    transitions = _read_transitions(trace, episode_index=args.episode_index)
    ass_path = output.with_suffix(".subtasks.ass")
    _write_ass(
        ass_path,
        transitions,
        fps=fps,
        width=output_width,
        height=output_height,
        duration=duration,
    )

    filter_graph = (
        f"scale={output_width}:{output_height}:flags=lanczos,"
        f"subtitles={ass_path.as_posix()}"
    )
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-i",
        str(video),
        "-vf",
        filter_graph,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "copy",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(f"Wrote {output}")
    print(f"Subtask transitions: {len(transitions)}")
    for frame_id, instruction in transitions:
        print(f"  trace frame {frame_id}: {instruction}")


if __name__ == "__main__":
    main()
