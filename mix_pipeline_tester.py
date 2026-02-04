#!/usr/bin/env python
"""Run analyzer + renderer end-to-end without DB/service dependencies."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import re
from typing import Any
import uuid

from atomix.analyzers.mix_analyzer import MixAnalyzer, MixAnalyzerGlobalHPSS
from atomix.renderers.mix_renderer import (
    DeterministicMixRenderer,
    RenderResult,
    TrackInput,
    VariableBpmMixRenderer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full mixing pipeline (analyze + render) on audio files in a directory."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing input audio tracks.",
    )
    parser.add_argument(
        "--analyzer-variant",
        default="local",
        choices=["local", "global_hpss"],
        help="Analyzer variant: local section HPSS or global full-track HPSS.",
    )
    parser.add_argument(
        "--renderer",
        default="deterministic",
        choices=["deterministic", "variable_bpm"],
        help="Renderer variant to use for audio rendering.",
    )
    parser.add_argument(
        "--analysis-sr",
        type=int,
        default=22050,
        help="Analyzer sample rate.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Analyzer hop length.",
    )
    parser.add_argument(
        "--fixed-prefix-count",
        type=int,
        default=0,
        help="Number of first tracks to treat as fixed prefix.",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON file with track metadata list (same length/order as discovered tracks).",
    )
    parser.add_argument(
        "--output-audio",
        default=None,
        help="Output mix audio path (default: ./mix_pipeline_outputs/rendered_mix_<variant>.wav).",
    )
    parser.add_argument(
        "--output-segments-json",
        default=None,
        help="Output JSON summary path (default: next to output audio).",
    )
    parser.add_argument(
        "--analysis-output-dir",
        default=None,
        help="Directory to save per-track analysis JSON files (default: ./mix_pipeline_outputs/analysis_<variant>).",
    )
    parser.add_argument(
        "--enable-timing-logs",
        action="store_true",
        help="Enable timing logs from analyzer + renderer.",
    )
    parser.add_argument(
        "--enable-renderer-debug-logs",
        action="store_true",
        help="Enable renderer debug decision logs.",
    )
    return parser.parse_args()


SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
    ".opus",
}


def _collect_tracks_from_dir(input_dir: Path) -> list[Path]:
    tracks = [
        p.resolve()
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    tracks.sort(key=lambda p: (p.name.lower(), str(p)))
    return tracks


def _load_metadata(path: str | None, track_paths: list[Path]) -> list[dict[str, Any]]:
    if path is None:
        return [{"song_name": p.stem, "artist_name": ""} for p in track_paths]

    metadata_path = Path(path).expanduser().resolve()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("metadata JSON must be a list")
    if len(payload) != len(track_paths):
        raise ValueError("metadata JSON length must match track count")

    out: list[dict[str, Any]] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"metadata[{i}] must be an object")
        song_name = str(item.get("song_name", track_paths[i].stem))
        artist_name = str(item.get("artist_name", ""))
        merged = dict(item)
        merged["song_name"] = song_name
        merged["artist_name"] = artist_name
        out.append(merged)
    return out


def _mime_to_suffix(mime: str) -> str:
    if mime in ("audio/mpeg", "audio/mp3"):
        return ".mp3"
    if mime == "audio/wav":
        return ".wav"
    return ".bin"


def _resolve_output_audio_path(output_audio: str | None, analyzer_variant: str, renderer_variant: str, mime: str) -> Path:
    suffix = _mime_to_suffix(mime)
    if output_audio:
        p = Path(output_audio).expanduser().resolve()
        if not p.suffix:
            p = p.with_suffix(suffix)
        return p
    return Path("mix_pipeline_outputs").resolve() / f"rendered_mix_{analyzer_variant}_{renderer_variant}{suffix}"


def _resolve_output_segments_path(output_segments_json: str | None, output_audio_path: Path) -> Path:
    if output_segments_json:
        return Path(output_segments_json).expanduser().resolve()
    return output_audio_path.with_name(f"{output_audio_path.stem}_segments.json")


def _resolve_analysis_output_dir(analysis_output_dir: str | None, variant: str) -> Path:
    if analysis_output_dir:
        return Path(analysis_output_dir).expanduser().resolve()
    return Path("mix_pipeline_outputs").resolve() / f"analysis_{variant}"


def _safe_filename_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "track"


async def _run_pipeline(
    track_paths: list[Path],
    metadata: list[dict[str, Any]],
    *,
    analyzer_variant: str,
    renderer_variant: str,
    analysis_sr: int,
    hop_length: int,
    fixed_prefix_count: int,
    enable_timing_logs: bool,
    enable_renderer_debug_logs: bool,
    analysis_output_dir: Path,
) -> tuple[list[TrackInput], RenderResult, list[str]]:
    analyzer_cls = MixAnalyzerGlobalHPSS if analyzer_variant == "global_hpss" else MixAnalyzer
    analyzer = analyzer_cls(
        analysis_sr=analysis_sr,
        hop_length=hop_length,
        enable_timing_logs=enable_timing_logs,
    )
    if renderer_variant == "variable_bpm":
        renderer = VariableBpmMixRenderer(
            enable_timing_logs=enable_timing_logs,
            enable_debug_logs=enable_renderer_debug_logs,
        )
    else:
        renderer = DeterministicMixRenderer(
            enable_timing_logs=enable_timing_logs,
            enable_debug_logs=enable_renderer_debug_logs,
        )

    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    track_inputs: list[TrackInput] = []
    analysis_paths: list[str] = []
    for i, path in enumerate(track_paths):
        analysis_result = await analyzer.analyze(str(path), metadata[i])
        analysis_path = analysis_output_dir / (
            f"{i:03d}_{_safe_filename_fragment(path.stem)}_analysis_{analyzer_variant}.json"
        )
        analysis_path.write_text(json.dumps(analysis_result.analysis_json, indent=2), encoding="utf-8")
        analysis_paths.append(str(analysis_path))
        # Deterministic ID based on input path + index.
        mix_item_id = uuid.uuid5(uuid.NAMESPACE_URL, f"mix-pipeline-tester:{i}:{path.as_posix()}")
        track_inputs.append(
            TrackInput(
                mix_item_id=mix_item_id,
                abs_path=str(path),
                metadata_json=analysis_result.metadata_json,
                analysis_json=analysis_result.analysis_json,
            )
        )

    prefix_n = max(0, min(fixed_prefix_count, len(track_inputs)))
    fixed_tracks = track_inputs[:prefix_n]
    render_result = await renderer.render(track_inputs, fixed_tracks)
    return track_inputs, render_result, analysis_paths


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    track_paths = _collect_tracks_from_dir(input_dir)
    if not track_paths:
        raise ValueError(
            f"No supported audio files found in {input_dir}. "
            f"Supported extensions: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    metadata = _load_metadata(args.metadata_json, track_paths)
    analysis_output_dir = _resolve_analysis_output_dir(args.analysis_output_dir, str(args.analyzer_variant))
    track_inputs, render_result, analysis_paths = asyncio.run(
        _run_pipeline(
            track_paths,
            metadata,
            analyzer_variant=str(args.analyzer_variant),
            renderer_variant=str(args.renderer),
            analysis_sr=int(args.analysis_sr),
            hop_length=int(args.hop_length),
            fixed_prefix_count=int(args.fixed_prefix_count),
            enable_timing_logs=bool(args.enable_timing_logs),
            enable_renderer_debug_logs=bool(args.enable_renderer_debug_logs),
            analysis_output_dir=analysis_output_dir,
        )
    )

    output_audio_path = _resolve_output_audio_path(
        output_audio=args.output_audio,
        analyzer_variant=str(args.analyzer_variant),
        renderer_variant=str(args.renderer),
        mime=render_result.mime,
    )
    output_segments_path = _resolve_output_segments_path(args.output_segments_json, output_audio_path)

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    output_audio_path.write_bytes(render_result.audio_bytes)

    segments_payload = [
        {
            "position": int(s.position),
            "mix_item_id": str(s.mix_item_id),
            "start_ms": int(s.start_ms),
            "end_ms": int(s.end_ms),
            "source_start_ms": int(s.source_start_ms),
            "song_name": s.song_name,
            "artist_name": s.artist_name,
        }
        for s in render_result.segments
    ]

    summary = {
        "analyzer_variant": str(args.analyzer_variant),
        "renderer_variant": str(args.renderer),
        "analysis_sr": int(args.analysis_sr),
        "hop_length": int(args.hop_length),
        "input_directory": str(input_dir),
        "input_tracks": [
            {
                "mix_item_id": str(t.mix_item_id),
                "abs_path": t.abs_path,
                "metadata_json": t.metadata_json,
                "analysis_json_path": analysis_paths[i],
            }
            for i, t in enumerate(track_inputs)
        ],
        "analysis_output_dir": str(analysis_output_dir),
        "fixed_prefix_count": int(max(0, min(args.fixed_prefix_count, len(track_inputs)))),
        "output_audio": {
            "path": str(output_audio_path),
            "mime": render_result.mime,
            "length_ms": int(render_result.length_ms),
            "bytes": len(render_result.audio_bytes),
        },
        "segments": segments_payload,
        "render_debug": render_result.debug or {},
    }

    output_segments_path.parent.mkdir(parents=True, exist_ok=True)
    output_segments_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Rendered mix: {output_audio_path}")
    print(f"[OK] Segment summary: {output_segments_path}")
    print(f"[OK] Per-track analysis JSONs: {analysis_output_dir}")
    print(f"[OK] Segments: {len(render_result.segments)} | Length: {render_result.length_ms} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
