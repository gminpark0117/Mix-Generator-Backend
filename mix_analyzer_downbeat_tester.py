#!/usr/bin/env python
"""Run MixAnalyzer end-to-end and save the full analysis_json output.

This is useful for profiling/validating the exact production analysis logic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from atomix.analyzers.mix_analyzer import MixAnalyzer, MixAnalyzerGlobalHPSS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MixAnalyzer end-to-end and write analysis_json to disk."
    )
    parser.add_argument("--input", required=True, help="Path to source audio track.")
    parser.add_argument(
        "--analysis-json",
        default=None,
        help="Output path for analysis_json (defaults to <input_stem>_mix_analyzer_analysis.json).",
    )
    parser.add_argument(
        "--output-wav",
        default=None,
        help="Optional output wav path with click overlay (defaults to <input_stem>_mix_analyzer_clicks.wav).",
    )
    parser.add_argument(
        "--overlay-mode",
        default="both",
        choices=["switch_in", "switch_out", "both"],
        help="Which section beat/downbeat times to overlay as clicks.",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional path to a metadata JSON file to pass through unchanged.",
    )
    parser.add_argument("--analysis-sr", type=int, default=22050, help="Analysis sample rate.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for analysis.")
    parser.add_argument(
        "--analyzer-variant",
        default="local",
        choices=["local", "global_hpss"],
        help="Analyzer variant to use: local (section HPSS) or global_hpss (full-track HPSS + CV beat source).",
    )
    meter_group = parser.add_mutually_exclusive_group()
    meter_group.add_argument(
        "--force-4-4",
        dest="force_beats_per_bar_4",
        action="store_true",
        help="Force meter inference to consider only 4 beats per bar (default).",
    )
    meter_group.add_argument(
        "--allow-non-4-4",
        dest="force_beats_per_bar_4",
        action="store_false",
        help="Allow meter inference to consider both 4/4 and 3/4.",
    )
    parser.set_defaults(force_beats_per_bar_4=True)
    parser.add_argument("--beat-gain", type=float, default=0.9, help="Gain for normal beat clicks.")
    parser.add_argument("--downbeat-gain", type=float, default=1.2, help="Gain for downbeat clicks.")
    return parser.parse_args()


def _synthesize_click(sr: int, freq_hz: float, gain: float, duration_s: float) -> np.ndarray:
    n_samples = max(1, int(round(duration_s * sr)))
    t = np.arange(n_samples, dtype=np.float32) / float(sr)
    env = np.exp(-24.0 * t)
    wave = np.sin(2.0 * np.pi * freq_hz * t)
    return (gain * env * wave).astype(np.float32)


def _overlay_clicks(
    audio: np.ndarray,
    sr: int,
    beats_s: list[float],
    downbeats_s: list[float],
    beat_gain: float,
    downbeat_gain: float,
) -> np.ndarray:
    if audio.ndim != 2:
        raise ValueError("Expected audio with shape (samples, channels).")
    out = audio.copy().astype(np.float32)
    beat_click = _synthesize_click(sr=sr, freq_hz=1300.0, gain=beat_gain, duration_s=0.05)
    downbeat_click = _synthesize_click(sr=sr, freq_hz=1900.0, gain=downbeat_gain, duration_s=0.08)
    downbeats = np.asarray(downbeats_s, dtype=np.float64)

    for beat_time in beats_s:
        start = int(round(float(beat_time) * sr))
        if start < 0 or start >= out.shape[0]:
            continue
        is_downbeat = bool(downbeats.size and np.min(np.abs(downbeats - beat_time)) <= 0.05)
        click = downbeat_click if is_downbeat else beat_click
        end = min(out.shape[0], start + click.shape[0])
        out[start:end, :] += click[: end - start, np.newaxis]

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out *= 0.99 / peak
    return out


def _collect_overlay_events(analysis_json: dict, overlay_mode: str) -> tuple[list[float], list[float]]:
    sections = analysis_json.get("sections") or {}

    def section_events(section_key: str) -> tuple[list[float], list[float]]:
        sec = sections.get(section_key) or {}
        rhythm = sec.get("rhythm") or {}
        beats_s = rhythm.get("beats_s") or []
        downbeats_s = rhythm.get("downbeats_s") or []
        return [float(b) for b in beats_s], [float(d) for d in downbeats_s]

    beats_all: list[float] = []
    downbeats_all: list[float] = []

    if overlay_mode in ("switch_in", "both"):
        b, d = section_events("switch_in")
        beats_all.extend(b)
        downbeats_all.extend(d)
    if overlay_mode in ("switch_out", "both"):
        b, d = section_events("switch_out")
        beats_all.extend(b)
        downbeats_all.extend(d)

    beats_all = sorted(set(round(b, 6) for b in beats_all))
    downbeats_all = sorted(set(round(d, 6) for d in downbeats_all))
    return beats_all, downbeats_all


def _fallback_overlay_events(input_path: Path, analyzer: MixAnalyzer) -> tuple[list[float], list[float]]:
    """Fallback if section-local rhythm is empty: infer full-track beats/downbeats."""
    try:
        y, sr = librosa.load(
            str(input_path),
            sr=analyzer.analysis_sr,
            mono=True,
            res_type="kaiser_fast",
        )
    except ModuleNotFoundError as exc:
        if "resampy" not in str(exc):
            raise
        print("[WARN] resampy not installed; fallback load uses soxr_hq.")
        y, sr = librosa.load(
            str(input_path),
            sr=analyzer.analysis_sr,
            mono=True,
            res_type="soxr_hq",
        )
    if len(y) == 0:
        return [], []

    # Prefer percussive onset for fallback beat inference.
    _, y_perc = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=analyzer.hop_length)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=analyzer.hop_length)
    beats_s = analyzer._beat_track_from_onset(onset_env, int(sr))

    if not beats_s:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=analyzer.hop_length)
        onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=analyzer.hop_length)
        beats_s = analyzer._beat_track_from_onset(onset_env, int(sr))

    if not beats_s:
        return [], []

    _, _, _, downbeats_s = analyzer._infer_meter_and_signature(
        beats_s=beats_s,
        onset_env=onset_env,
        onset_times=onset_times,
    )
    return [float(b) for b in beats_s], [float(d) for d in downbeats_s]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    variant_tag = str(args.analyzer_variant).lower()
    analysis_json_path = (
        Path(args.analysis_json).expanduser().resolve()
        if args.analysis_json
        else input_path.with_name(f"{input_path.stem}_mix_analyzer_analysis_{variant_tag}.json")
    )
    output_wav_path = (
        Path(args.output_wav).expanduser().resolve()
        if args.output_wav
        else input_path.with_name(f"{input_path.stem}_mix_analyzer_clicks_{variant_tag}.wav")
    )

    if args.metadata_json:
        metadata_path = Path(args.metadata_json).expanduser().resolve()
        metadata_json = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata_json = {}

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    analyzer_cls = MixAnalyzerGlobalHPSS if args.analyzer_variant == "global_hpss" else MixAnalyzer
    analyzer = analyzer_cls(
        analysis_sr=int(args.analysis_sr),
        hop_length=int(args.hop_length),
        force_beats_per_bar_4=bool(args.force_beats_per_bar_4),
        enable_timing_logs=True,
    )
    result = asyncio.run(analyzer.analyze(str(input_path), metadata_json))

    analysis_json_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_json_path.write_text(json.dumps(result.analysis_json, indent=2), encoding="utf-8")

    # Optional click overlay, using beat/downbeat times as inferred by MixAnalyzer within its sections.
    beats_s, downbeats_s = _collect_overlay_events(result.analysis_json, overlay_mode=str(args.overlay_mode))
    if not beats_s:
        print("[WARN] No section beats found for overlay; falling back to full-track beat inference.")
        beats_s, downbeats_s = _fallback_overlay_events(input_path, analyzer)

    if beats_s:
        audio, sr_out = sf.read(str(input_path), dtype="float32", always_2d=True)

        rendered = _overlay_clicks(
            audio=audio,
            sr=int(sr_out),
            beats_s=beats_s,
            downbeats_s=downbeats_s,
            beat_gain=float(args.beat_gain),
            downbeat_gain=float(args.downbeat_gain),
        )

        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav_path), rendered, int(sr_out))
    else:
        print("[WARN] No beats found after fallback; skipping overlay wav output.")

    print(f"[OK] analysis_json saved: {analysis_json_path}")
    if beats_s:
        print(f"[OK] overlay wav saved: {output_wav_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
