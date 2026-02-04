from __future__ import annotations

import asyncio
import importlib.util
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import librosa
else:
    try:
        import numpy as np
        import librosa
    except Exception:  # pragma: no cover - handled at runtime in analyze
        np = None
        librosa = None


@dataclass(frozen=True)
class AnalysisResult:
    """Result from analyzing a track."""

    metadata_json: dict
    analysis_json: dict


class MixAnalyzer:
    """
    Librosa-only analyzer that emphasizes section-local (switch-in/out) descriptors.
    """
    USE_GLOBAL_HPSS = False
    PREFER_LOCAL_BEATS = True
    VARIANT_NAME = "local_hpss_sections"

    def __init__(
        self,
        *,
        analysis_sr: int = 22050,
        hop_length: int = 512,
        phase_bins_per_bar: int = 96,
        tempogram_peaks: int = 5,
        energy_curve_points: int = 60,
        switch_region_span_s: float = 30.0,
        boundary_scan_seconds: float = 30.0,
        enable_timing_logs: bool = False,
    ) -> None:
        self.analysis_sr = analysis_sr
        self.hop_length = hop_length
        self.phase_bins_per_bar = phase_bins_per_bar
        self.tempogram_peaks = tempogram_peaks
        self.energy_curve_points = energy_curve_points
        self.switch_region_span_s = max(4.0, float(switch_region_span_s))
        self.boundary_scan_seconds = max(0.0, float(boundary_scan_seconds))
        self.audio_load_res_type = "soxr_hq" if importlib.util.find_spec("soxr") is not None else "kaiser_fast"
        self.enable_timing_logs = enable_timing_logs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def analyze(self, abs_path: str, metadata_json: dict) -> AnalysisResult:
        """
        Analyze a track file.

        Args:
            abs_path: absolute path to audio file
            metadata_json: track metadata (song_name, artist_name, etc.)

        Returns:
            AnalysisResult with metadata_json and analysis_json dict
        """
        return await asyncio.to_thread(self._analyze_sync, abs_path, metadata_json)

    def _analyze_sync(self, abs_path: str, metadata_json: dict) -> AnalysisResult:
        total_start = time.perf_counter()
        debug: dict[str, Any] = {
            "params": {
                "analysis_sr": self.analysis_sr,
                "hop_length": self.hop_length,
                "phase_bins_per_bar": self.phase_bins_per_bar,
                "tempogram_peaks": self.tempogram_peaks,
                "energy_curve_points": self.energy_curve_points,
                "switch_region_span_s": self.switch_region_span_s,
                "boundary_scan_seconds": self.boundary_scan_seconds,
                "audio_load_res_type": self.audio_load_res_type,
                "enable_timing_logs": self.enable_timing_logs,
                "variant_name": self.VARIANT_NAME,
                "use_global_hpss": self.USE_GLOBAL_HPSS,
                "beat_selection_mode": "prefer_local" if self.PREFER_LOCAL_BEATS else "cv_compare",
            },
            "fallbacks": [],
            "decisions": {},
            "errors": [],
            "timing_s": {},
        }

        if np is None or librosa is None:
            debug["errors"].append("audio_deps_missing")
            self._record_timing(debug, "total", total_start)
            return AnalysisResult(metadata_json=metadata_json, analysis_json=self._empty_analysis(debug))

        boundary_scan_active = bool((not self.USE_GLOBAL_HPSS) and self.boundary_scan_seconds > 0.0)
        y = None
        sr = int(self.analysis_sr)
        duration_s = 0.0
        boundary_segments: list[tuple[float, float, Any]] = []

        step_start = time.perf_counter()
        try:
            if boundary_scan_active:
                boundary_segments, sr, duration_s = self._load_boundary_segments(abs_path, debug)
                if boundary_segments:
                    debug["decisions"]["boundary_scan_active"] = True
                    debug["decisions"]["boundary_segments"] = [
                        {"start_s": float(s), "end_s": float(e)} for s, e, _ in boundary_segments
                    ]
                else:
                    y, sr = self._load_audio(abs_path, debug)
                    duration_s = float(librosa.get_duration(y=y, sr=sr))
                    boundary_segments = [(0.0, float(duration_s), y)]
                    debug["fallbacks"].append("boundary_scan_fallback_full_load")
            else:
                y, sr = self._load_audio(abs_path, debug)
                duration_s = float(librosa.get_duration(y=y, sr=sr))
        except Exception as exc:  # pragma: no cover - dependent on file IO
            debug["errors"].append(f"audio_load_failed: {exc}")
            self._record_timing(debug, "load_audio", step_start)
            self._record_timing(debug, "total", total_start)
            return AnalysisResult(metadata_json=metadata_json, analysis_json=self._empty_analysis(debug))
        self._record_timing(debug, "load_audio", step_start)

        y_harm_full = None
        y_perc_full = None
        if self.USE_GLOBAL_HPSS:
            step_start = time.perf_counter()
            y_harm_full, y_perc_full = librosa.effects.hpss(y)
            self._record_timing(debug, "hpss_global", step_start)

        segment_onsets: list[tuple[float, Any, Any]] = []
        step_start = time.perf_counter()
        if boundary_scan_active:
            onset_env_parts: list[Any] = []
            onset_time_parts: list[Any] = []
            for seg_start_s, _seg_end_s, seg_audio in boundary_segments:
                onset_seg = librosa.onset.onset_strength(y=seg_audio, sr=sr, hop_length=self.hop_length)
                onset_times_seg = (
                    librosa.frames_to_time(
                        np.arange(len(onset_seg)),
                        sr=sr,
                        hop_length=self.hop_length,
                    )
                    + float(seg_start_s)
                )
                segment_onsets.append((float(seg_start_s), onset_seg, onset_times_seg))
                onset_env_parts.append(onset_seg)
                onset_time_parts.append(onset_times_seg)
            onset_env_full = (
                np.concatenate(onset_env_parts)
                if onset_env_parts
                else np.asarray([], dtype=np.float32)
            )
            onset_times_full = (
                np.concatenate(onset_time_parts)
                if onset_time_parts
                else np.asarray([], dtype=np.float32)
            )
        else:
            onset_source = y_perc_full if y_perc_full is not None else y
            onset_env_full = librosa.onset.onset_strength(y=onset_source, sr=sr, hop_length=self.hop_length)
            onset_times_full = librosa.frames_to_time(
                np.arange(len(onset_env_full)),
                sr=sr,
                hop_length=self.hop_length,
            )
        self._record_timing(debug, "onset_full", step_start)

        step_start = time.perf_counter()
        if boundary_scan_active:
            beats_s_full: list[float] = []
            for seg_start_s, onset_seg, _ in segment_onsets:
                beats_seg = self._beat_track_from_onset(onset_seg, sr)
                beats_s_full.extend([float(seg_start_s + b) for b in beats_seg])
            beats_s_full = sorted(self._clean_floats(beats_s_full))
        else:
            beats_s_full = self._beat_track_from_onset(onset_env_full, sr)
        global_bpm_guess, _ = self._tempo_from_beats(beats_s_full)
        self._record_timing(debug, "beats_full", step_start)

        step_start = time.perf_counter()
        if boundary_scan_active:
            rms_parts: list[Any] = []
            rms_times_parts: list[Any] = []
            flux_parts: list[Any] = []
            flux_times_parts: list[Any] = []
            chroma_parts: list[Any] = []
            chroma_times_parts: list[Any] = []
            for seg_start_s, _seg_end_s, seg_audio in boundary_segments:
                rms_seg, rms_times_seg = self._rms_curve(seg_audio, sr)
                flux_seg, flux_times_seg = self._spectral_flux(seg_audio, sr)
                chroma_seg, chroma_times_seg = self._chroma_flux(seg_audio, sr)
                rms_parts.append(rms_seg)
                rms_times_parts.append(rms_times_seg + float(seg_start_s))
                flux_parts.append(flux_seg)
                flux_times_parts.append(flux_times_seg + float(seg_start_s))
                chroma_parts.append(chroma_seg)
                chroma_times_parts.append(chroma_times_seg + float(seg_start_s))

            rms_full = np.concatenate(rms_parts) if rms_parts else np.asarray([], dtype=np.float32)
            rms_times_full = (
                np.concatenate(rms_times_parts)
                if rms_times_parts
                else np.asarray([], dtype=np.float32)
            )
            flux_full = np.concatenate(flux_parts) if flux_parts else np.asarray([], dtype=np.float32)
            flux_times_full = (
                np.concatenate(flux_times_parts)
                if flux_times_parts
                else np.asarray([], dtype=np.float32)
            )
            chroma_flux_full = (
                np.concatenate(chroma_parts)
                if chroma_parts
                else np.asarray([], dtype=np.float32)
            )
            chroma_times_full = (
                np.concatenate(chroma_times_parts)
                if chroma_times_parts
                else np.asarray([], dtype=np.float32)
            )
        else:
            rms_full, rms_times_full = self._rms_curve(y, sr)
            flux_source = y_perc_full if y_perc_full is not None else y
            chroma_source = y_harm_full if y_harm_full is not None else y
            flux_full, flux_times_full = self._spectral_flux(flux_source, sr)
            chroma_flux_full, chroma_times_full = self._chroma_flux(chroma_source, sr)
        self._record_timing(debug, "global_curves", step_start)

        region_span = float(self.switch_region_span_s)
        intro_region = (0.0, min(duration_s, region_span))
        outro_region = (max(0.0, duration_s - region_span), duration_s)

        step_start = time.perf_counter()
        switch_in_window = self._select_switch_window(
            label="switch_in",
            region=intro_region,
            beats_s=beats_s_full,
            onset_env=onset_env_full,
            onset_times=onset_times_full,
            rms=rms_full,
            rms_times=rms_times_full,
            spectral_flux=flux_full,
            spectral_flux_times=flux_times_full,
            chroma_flux=chroma_flux_full,
            chroma_flux_times=chroma_times_full,
            duration_s=duration_s,
            debug=debug,
        )
        self._record_timing(debug, "select_switch_in_window", step_start)

        step_start = time.perf_counter()
        switch_out_window = self._select_switch_window(
            label="switch_out",
            region=outro_region,
            beats_s=beats_s_full,
            onset_env=onset_env_full,
            onset_times=onset_times_full,
            rms=rms_full,
            rms_times=rms_times_full,
            spectral_flux=flux_full,
            spectral_flux_times=flux_times_full,
            chroma_flux=chroma_flux_full,
            chroma_flux_times=chroma_times_full,
            duration_s=duration_s,
            debug=debug,
        )
        self._record_timing(debug, "select_switch_out_window", step_start)

        step_start = time.perf_counter()
        switch_in = self._analyze_section(
            label="switch_in",
            start_s=switch_in_window["start_s"],
            end_s=switch_in_window["end_s"],
            y=y,
            boundary_segments=boundary_segments if boundary_scan_active else None,
            y_harm_full=y_harm_full,
            y_perc_full=y_perc_full,
            sr=sr,
            beats_s_full=beats_s_full,
            quality=switch_in_window["quality"],
            debug=debug,
        )
        self._record_timing(debug, "analyze_switch_in", step_start)

        step_start = time.perf_counter()
        switch_out = self._analyze_section(
            label="switch_out",
            start_s=switch_out_window["start_s"],
            end_s=switch_out_window["end_s"],
            y=y,
            boundary_segments=boundary_segments if boundary_scan_active else None,
            y_harm_full=y_harm_full,
            y_perc_full=y_perc_full,
            sr=sr,
            beats_s_full=beats_s_full,
            quality=switch_out_window["quality"],
            debug=debug,
        )
        self._record_timing(debug, "analyze_switch_out", step_start)

        step_start = time.perf_counter()
        analysis_json = {
            "schema_version": 2,
            "track": {
                "duration_s": float(duration_s),
                "sr": int(sr),
                "hop_length": int(self.hop_length),
                "overview": {
                    "global_bpm_guess": global_bpm_guess,
                },
            },
            "sections": {
                "switch_in": switch_in,
                "switch_out": switch_out,
            },
            "debug": debug,
        }
        self._record_timing(debug, "assemble_output", step_start)
        self._record_timing(debug, "total", total_start)

        return AnalysisResult(metadata_json=metadata_json, analysis_json=analysis_json)

    def _probe_duration(self, abs_path: str, debug: dict[str, Any]) -> float | None:
        try:
            duration_s = float(librosa.get_duration(path=abs_path))
            if math.isfinite(duration_s) and duration_s > 0:
                return duration_s
        except Exception:
            debug["fallbacks"].append("duration_probe_failed")
        return None

    def _load_boundary_segments(
        self,
        abs_path: str,
        debug: dict[str, Any],
    ) -> tuple[list[tuple[float, float, Any]], int, float]:
        duration_s = self._probe_duration(abs_path, debug)
        if duration_s is None:
            return [], int(self.analysis_sr), 0.0

        span = float(min(self.boundary_scan_seconds, duration_s))
        intro_end_s = span
        outro_start_s = max(0.0, duration_s - span)

        ranges: list[tuple[float, float]] = [(0.0, intro_end_s)]
        if outro_start_s > intro_end_s + 1e-6:
            ranges.append((outro_start_s, duration_s))

        segments: list[tuple[float, float, Any]] = []
        sr_out: int | None = None
        for start_s, end_s in ranges:
            clip_duration = max(0.0, float(end_s - start_s))
            if clip_duration <= 0.0:
                continue
            clip, sr = self._load_audio(abs_path, debug, offset_s=start_s, duration_s=clip_duration)
            sr_out = int(sr if sr_out is None else sr_out)
            segments.append((float(start_s), float(end_s), clip))

        return segments, int(sr_out or self.analysis_sr), float(duration_s)

    def _load_audio(
        self,
        abs_path: str,
        debug: dict[str, Any],
        *,
        offset_s: float = 0.0,
        duration_s: float | None = None,
    ) -> tuple[Any, int]:
        res_types: list[str] = []
        preferred = str(self.audio_load_res_type or "").strip()
        if preferred:
            res_types.append(preferred)
        if "soxr_hq" not in res_types:
            res_types.append("soxr_hq")
        if "kaiser_fast" not in res_types:
            res_types.append("kaiser_fast")

        last_exc: Exception | None = None
        for idx, res_type in enumerate(res_types):
            try:
                y, sr = librosa.load(
                    abs_path,
                    sr=self.analysis_sr,
                    mono=True,
                    offset=max(0.0, float(offset_s)),
                    duration=(None if duration_s is None else max(0.0, float(duration_s))),
                    res_type=res_type,
                )
                if idx > 0:
                    debug["fallbacks"].append(f"audio_load_res_type_fallback_{res_type}")
                return y, int(sr)
            except ModuleNotFoundError as exc:
                last_exc = exc
                msg = str(exc)
                if res_type.startswith("soxr_") and "soxr" in msg:
                    continue
                if res_type.startswith("kaiser") and "resampy" in msg:
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("audio_load_failed_no_resampler")

    def _record_timing(self, debug: dict[str, Any], label: str, start: float) -> None:
        elapsed = float(time.perf_counter() - start)
        timing = debug.setdefault("timing_s", {})
        timing[label] = elapsed
        if self.enable_timing_logs:
            self.logger.info("Timing %s: %.3fs", label, elapsed)

    def _beat_track_from_onset(self, onset_env: Any, sr: int) -> list[float]:
        if onset_env is None or len(onset_env) == 0:
            return []
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
        )
        beats_s = librosa.frames_to_time(
            beat_frames,
            sr=sr,
            hop_length=self.hop_length,
        ).tolist()
        return self._clean_floats(beats_s)

    def _select_switch_window(
        self,
        *,
        label: str,
        region: tuple[float, float],
        beats_s: list[float],
        onset_env: Any,
        onset_times: Any,
        rms: Any,
        rms_times: Any,
        spectral_flux: Any,
        spectral_flux_times: Any,
        chroma_flux: Any,
        chroma_flux_times: Any,
        duration_s: float,
        debug: dict[str, Any],
    ) -> dict[str, Any]:
        region_start, region_end = region
        region_start = float(max(0.0, region_start))
        region_end = float(min(duration_s, region_end))

        window_beats = 16
        step_beats = 4

        candidates: list[tuple[float, float]] = []
        if len(beats_s) >= window_beats + 1:
            for i in range(0, len(beats_s) - window_beats - 1, step_beats):
                start_s = beats_s[i]
                end_s = beats_s[i + window_beats]
                if start_s < region_start or end_s > region_end:
                    continue
                candidates.append((float(start_s), float(end_s)))

        if not candidates:
            debug["fallbacks"].append(f"{label}_time_window_fallback")
            window_len = min(16.0, max(4.0, region_end - region_start))
            start_s = region_start
            end_s = min(region_end, start_s + window_len)
            candidates.append((float(start_s), float(end_s)))

        best_score = -1e9
        best_window = candidates[0]
        best_quality: dict[str, Any] = {"score": 0.0, "components": {}}

        for start_s, end_s in candidates:
            quality = self._score_window(
                start_s=start_s,
                end_s=end_s,
                beats_s=beats_s,
                onset_env=onset_env,
                onset_times=onset_times,
                rms=rms,
                rms_times=rms_times,
                spectral_flux=spectral_flux,
                spectral_flux_times=spectral_flux_times,
                chroma_flux=chroma_flux,
                chroma_flux_times=chroma_flux_times,
            )
            if quality["score"] > best_score:
                best_score = float(quality["score"])
                best_window = (start_s, end_s)
                best_quality = quality

        debug["decisions"][label] = {
            "window_start_s": float(best_window[0]),
            "window_end_s": float(best_window[1]),
            "quality": best_quality,
        }

        return {
            "start_s": float(best_window[0]),
            "end_s": float(best_window[1]),
            "quality": best_quality,
        }

    def _score_window(
        self,
        *,
        start_s: float,
        end_s: float,
        beats_s: list[float],
        onset_env: Any,
        onset_times: Any,
        rms: Any,
        rms_times: Any,
        spectral_flux: Any,
        spectral_flux_times: Any,
        chroma_flux: Any,
        chroma_flux_times: Any,
    ) -> dict[str, Any]:
        beats_win = [b for b in beats_s if start_s <= b <= end_s]
        _, beat_stats = self._tempo_from_beats(beats_win)
        tempo_stability = 1.0 / (1.0 + beat_stats.get("cv", 1.0))

        onset_mean = self._mean_in_window(onset_env, onset_times, start_s, end_s)
        onset_global_mean = float(np.mean(onset_env)) if onset_env is not None and len(onset_env) else 1.0
        onset_clarity = onset_mean / (onset_global_mean + 1e-9)

        rms_mean = self._mean_in_window(rms, rms_times, start_s, end_s)
        rms_slope = self._slope_in_window(rms, rms_times, start_s, end_s)
        energy_ramp = max(0.0, rms_slope) / (rms_mean + 1e-9) if rms_mean > 0 else 0.0

        spectral_flux_mean = self._mean_in_window(
            spectral_flux,
            spectral_flux_times,
            start_s,
            end_s,
        )
        spectral_flux_global = (
            float(np.mean(spectral_flux)) if spectral_flux is not None and len(spectral_flux) else 1.0
        )
        spectral_flux_norm = spectral_flux_mean / (spectral_flux_global + 1e-9)

        chroma_flux_mean = self._mean_in_window(
            chroma_flux,
            chroma_flux_times,
            start_s,
            end_s,
        )
        chroma_flux_global = (
            float(np.mean(chroma_flux)) if chroma_flux is not None and len(chroma_flux) else 1.0
        )
        chroma_flux_norm = chroma_flux_mean / (chroma_flux_global + 1e-9)

        score = (
            1.0 * onset_clarity
            + 1.0 * tempo_stability
            - 0.7 * energy_ramp
            - 0.5 * spectral_flux_norm
            - 0.5 * chroma_flux_norm
        )

        return {
            "score": float(score),
            "components": {
                "onset_clarity": float(onset_clarity),
                "tempo_stability": float(tempo_stability),
                "energy_ramp": float(energy_ramp),
                "spectral_flux": float(spectral_flux_norm),
                "chroma_flux": float(chroma_flux_norm),
            },
        }

    def _slice_from_boundary_segments(
        self,
        *,
        boundary_segments: list[tuple[float, float, Any]],
        start_s: float,
        end_s: float,
        sr: int,
    ) -> Any:
        for seg_start_s, seg_end_s, seg_audio in boundary_segments:
            if start_s >= (seg_start_s - 1e-6) and end_s <= (seg_end_s + 1e-6):
                rel_start = max(0.0, float(start_s - seg_start_s))
                rel_end = max(rel_start, float(end_s - seg_start_s))
                start_idx = int(rel_start * sr)
                end_idx = int(rel_end * sr)
                return seg_audio[start_idx:end_idx]

        pieces: list[Any] = []
        for seg_start_s, seg_end_s, seg_audio in boundary_segments:
            clip_start = max(float(start_s), float(seg_start_s))
            clip_end = min(float(end_s), float(seg_end_s))
            if clip_end <= clip_start:
                continue
            rel_start = clip_start - float(seg_start_s)
            rel_end = clip_end - float(seg_start_s)
            start_idx = int(rel_start * sr)
            end_idx = int(rel_end * sr)
            if end_idx > start_idx:
                pieces.append(seg_audio[start_idx:end_idx])
        if pieces:
            return np.concatenate(pieces)
        return np.asarray([], dtype=np.float32)

    def _analyze_section(
        self,
        *,
        label: str,
        start_s: float,
        end_s: float,
        y: Any | None,
        boundary_segments: list[tuple[float, float, Any]] | None,
        y_harm_full: Any,
        y_perc_full: Any,
        sr: int,
        beats_s_full: list[float],
        quality: dict[str, Any],
        debug: dict[str, Any],
    ) -> dict[str, Any]:
        section_total_start = time.perf_counter()
        timing_prefix = f"section.{label}"
        if end_s <= start_s:
            debug["fallbacks"].append(f"{label}_empty_window")
            self._record_timing(debug, f"{timing_prefix}.total", section_total_start)
            return {
                "start_s": float(start_s),
                "end_s": float(end_s),
                "quality": quality,
                "rhythm": {},
                "harmony": {},
                "timbre": {},
                "spectral": {},
                "energy": {},
            }

        if y is not None:
            start_idx = int(start_s * sr)
            end_idx = int(end_s * sr)
            y_section = y[start_idx:end_idx]
        elif boundary_segments:
            y_section = self._slice_from_boundary_segments(
                boundary_segments=boundary_segments,
                start_s=start_s,
                end_s=end_s,
                sr=sr,
            )
            start_idx = int(start_s * sr)
            end_idx = int(end_s * sr)
        else:
            y_section = np.asarray([], dtype=np.float32)
            start_idx = int(start_s * sr)
            end_idx = int(end_s * sr)

        if len(y_section) == 0:
            debug["fallbacks"].append(f"{label}_audio_slice_empty")
            self._record_timing(debug, f"{timing_prefix}.total", section_total_start)
            return {
                "start_s": float(start_s),
                "end_s": float(end_s),
                "quality": quality,
                "rhythm": {},
                "harmony": {},
                "timbre": {},
                "spectral": {},
                "energy": {},
            }

        step_start = time.perf_counter()
        if self.USE_GLOBAL_HPSS and y_harm_full is not None and y_perc_full is not None:
            y_harm_section = y_harm_full[start_idx:end_idx]
            y_perc_section = y_perc_full[start_idx:end_idx]
            self._record_timing(debug, f"{timing_prefix}.hpss_slice_global", step_start)
        else:
            y_harm_section, y_perc_section = librosa.effects.hpss(y_section)
            self._record_timing(debug, f"{timing_prefix}.hpss_section", step_start)

        step_start = time.perf_counter()
        onset_env_sec = librosa.onset.onset_strength(
            y=y_perc_section,
            sr=sr,
            hop_length=self.hop_length,
        )
        onset_times_sec = (
            librosa.frames_to_time(
                np.arange(len(onset_env_sec)),
                sr=sr,
                hop_length=self.hop_length,
            )
            + start_s
        )
        self._record_timing(debug, f"{timing_prefix}.onset", step_start)

        step_start = time.perf_counter()
        beats_local = self._beat_track_from_onset(onset_env_sec, sr)
        beats_local = [b + start_s for b in beats_local]
        beats_global = [b for b in beats_s_full if start_s <= b <= end_s]

        beats_s, beat_source = self._choose_beats(beats_local, beats_global)
        if beat_source != "local":
            debug["fallbacks"].append(f"{label}_beats_from_{beat_source}")
        self._record_timing(debug, f"{timing_prefix}.beats", step_start)

        step_start = time.perf_counter()
        bpm_local, beat_stats = self._tempo_from_beats(beats_s)
        bpm_candidates = self._tempo_candidates(bpm_local)
        self._record_timing(debug, f"{timing_prefix}.tempo_stats", step_start)

        step_start = time.perf_counter()
        meter_info, signature, beat_positions, downbeats_s = self._infer_meter_and_signature(
            beats_s=beats_s,
            onset_env=onset_env_sec,
            onset_times=onset_times_sec,
        )
        self._record_timing(debug, f"{timing_prefix}.meter_signature", step_start)

        step_start = time.perf_counter()
        harmony = self._harmony_features(y=y_harm_section, sr=sr)
        self._record_timing(debug, f"{timing_prefix}.harmony", step_start)
        step_start = time.perf_counter()
        timbre = self._timbre_features(y=y_section, sr=sr)
        self._record_timing(debug, f"{timing_prefix}.timbre", step_start)
        step_start = time.perf_counter()
        spectral = self._spectral_features(y=y_section, sr=sr)
        self._record_timing(debug, f"{timing_prefix}.spectral", step_start)

        rhythm = {
            "beats_s": self._clean_floats(beats_s),
            "bpm_local": bpm_local,
            "bpm_candidates": bpm_candidates,
            "meter": meter_info,
            "downbeats_s": self._clean_floats(downbeats_s),
            "beat_positions": beat_positions,
            "beat_interval_s": beat_stats,
            "signature": signature,
        }

        section_result = {
            "start_s": float(start_s),
            "end_s": float(end_s),
            "quality": quality,
            "rhythm": rhythm,
            "harmony": harmony,
            "timbre": timbre,
            "spectral": spectral,
            "energy": {},
        }
        self._record_timing(debug, f"{timing_prefix}.total", section_total_start)
        return section_result


    def _empty_analysis(self, debug: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "track": {
                "duration_s": 0.0,
                "sr": self.analysis_sr,
                "hop_length": self.hop_length,
                "overview": {},
            },
            "sections": {
                "switch_in": {
                    "start_s": 0.0,
                    "end_s": 0.0,
                    "quality": {},
                    "rhythm": {},
                    "harmony": {},
                    "timbre": {},
                    "spectral": {},
                    "energy": {},
                },
                "switch_out": {
                    "start_s": 0.0,
                    "end_s": 0.0,
                    "quality": {},
                    "rhythm": {},
                    "harmony": {},
                    "timbre": {},
                    "spectral": {},
                    "energy": {},
                },
            },
            "debug": debug,
        }

    def _choose_beats(self, beats_local: list[float], beats_global: list[float]) -> tuple[list[float], str]:
        if self.PREFER_LOCAL_BEATS:
            if len(beats_local) >= 3:
                return beats_local, "local"
            if len(beats_global) >= 3:
                return beats_global, "global"
            return beats_local if beats_local else beats_global, "insufficient"

        # Previous behavior: choose source by interval stability (CV).
        if len(beats_local) < 3 and len(beats_global) < 3:
            return beats_local if beats_local else beats_global, "insufficient"
        cv_local = self._interval_cv(beats_local)
        cv_global = self._interval_cv(beats_global)
        if cv_local <= cv_global:
            return beats_local, "local"
        return beats_global, "global"

    def _interval_cv(self, beats_s: list[float]) -> float:
        if len(beats_s) < 3:
            return 9.9
        intervals = np.diff(np.asarray(beats_s))
        intervals = intervals[intervals > 0]
        if len(intervals) == 0:
            return 9.9
        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))
        return float(std_interval / mean_interval) if mean_interval > 0 else 9.9

    def _infer_meter_and_signature(
        self,
        *,
        beats_s: list[float],
        onset_env: Any,
        onset_times: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int], list[float]]:
        if len(beats_s) < 2 or onset_env is None or len(onset_env) == 0:
            meter = {"beats_per_bar": 4, "note_value": 4, "confidence": 0.0, "assumed": True}
            signature = {
                "onset_source": "hpss_percussive",
                "periodicity": {"tempogram_top_peaks": [], "tempogram_profile_ds": []},
                "phase": {
                    "phase_bins_per_bar": self.phase_bins_per_bar,
                    "pattern": [],
                    "fft_mag": [],
                    "stability": {"bar_to_bar_var": 0.0, "n_bars_used": 0},
                    "tempo_scale_used": 1.0,
                    "bar_offset_used": 0,
                },
            }
            beat_positions = [1 for _ in beats_s]
            downbeats_s = beats_s[:1] if beats_s else []
            return meter, signature, beat_positions, downbeats_s

        candidate_scores: list[tuple[float, dict[str, Any]]] = []
        for tempo_scale in (0.5, 1.0, 2.0):
            beats_scaled = self._scale_beats(beats_s, tempo_scale)
            if len(beats_scaled) < 4:
                continue
            for beats_per_bar in (4, 3):
                phase = self._phase_from_beats(
                    beats_s=beats_scaled,
                    beats_per_bar=beats_per_bar,
                    onset_env=onset_env,
                    onset_times=onset_times,
                    phase_bins=self.phase_bins_per_bar,
                )
                candidate_scores.append(
                    (
                        phase["score"],
                        {
                            "tempo_scale": tempo_scale,
                            "beats_per_bar": beats_per_bar,
                            **phase,
                        },
                    )
                )

        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        best = candidate_scores[0][1] if candidate_scores else None
        second_score = candidate_scores[1][0] if len(candidate_scores) > 1 else None

        if best is None:
            meter = {"beats_per_bar": 4, "note_value": 4, "confidence": 0.0, "assumed": True}
            beat_positions = [1 for _ in beats_s]
            downbeats_s = beats_s[:1] if beats_s else []
            signature = self._signature_from_onset(onset_env, beats_s, meter, phase=None)
            return meter, signature, beat_positions, downbeats_s

        confidence = 0.2
        if second_score is not None and abs(best["score"]) > 1e-9:
            confidence = max(0.0, min(1.0, (best["score"] - second_score) / abs(best["score"])))

        meter = {
            "beats_per_bar": int(best["beats_per_bar"]),
            "note_value": 4,
            "confidence": float(confidence),
            "assumed": confidence < 0.5,
        }

        beat_positions = self._build_beat_positions(len(beats_s), meter["beats_per_bar"], best["bar_offset"])
        downbeats_s = [b for b, pos in zip(beats_s, beat_positions) if pos == 1]

        signature = self._signature_from_onset(
            onset_env,
            beats_s,
            meter,
            phase={
                "pattern": best["pattern"],
                "fft_mag": best["fft_mag"],
                "bar_to_bar_var": best["bar_to_bar_var"],
                "n_bars_used": best["n_bars_used"],
                "tempo_scale_used": best["tempo_scale"],
                "bar_offset_used": best["bar_offset"],
            },
        )

        return meter, signature, beat_positions, downbeats_s

    def _phase_from_beats(
        self,
        *,
        beats_s: list[float],
        beats_per_bar: int,
        onset_env: Any,
        onset_times: Any,
        phase_bins: int,
    ) -> dict[str, Any]:
        best_score = -1e9
        best_pattern: list[float] = []
        best_fft: list[float] = []
        best_var = 0.0
        best_offset = 0
        best_bars = 0

        for offset in range(beats_per_bar):
            downbeat_indices = [i for i in range(len(beats_s)) if (i - offset) % beats_per_bar == 0]
            if len(downbeat_indices) < 2:
                continue
            bar_patterns: list[np.ndarray] = []
            for a, b in zip(downbeat_indices[:-1], downbeat_indices[1:]):
                bar_start = beats_s[a]
                bar_end = beats_s[b]
                pattern = self._resample_onset_pattern(
                    onset_env=onset_env,
                    onset_times=onset_times,
                    start_s=bar_start,
                    end_s=bar_end,
                    bins=phase_bins,
                )
                if pattern is not None:
                    bar_patterns.append(pattern)
            if not bar_patterns:
                continue

            bars_matrix = np.vstack(bar_patterns)
            pattern_med = np.median(bars_matrix, axis=0)
            var = float(np.mean(np.var(bars_matrix, axis=0)))

            downbeat_strength = self._mean_at_times(
                onset_env,
                onset_times,
                [beats_s[i] for i in downbeat_indices],
            )
            score = downbeat_strength - var

            if score > best_score:
                best_score = score
                best_pattern = self._normalize_pattern(pattern_med)
                best_fft = self._fft_mag(best_pattern, n=16)
                best_var = var
                best_offset = offset
                best_bars = len(bar_patterns)

        return {
            "score": float(best_score),
            "pattern": best_pattern,
            "fft_mag": best_fft,
            "bar_to_bar_var": float(best_var),
            "n_bars_used": int(best_bars),
            "bar_offset": int(best_offset),
        }

    def _signature_from_onset(
        self,
        onset_env: Any,
        beats_s: list[float],
        meter: dict[str, Any],
        phase: dict[str, Any] | None,
    ) -> dict[str, Any]:
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=self.analysis_sr,
            hop_length=self.hop_length,
        )
        tempogram_mean = np.mean(tempogram, axis=1) if tempogram.size else np.array([])
        tempo_bins = librosa.tempo_frequencies(
            tempogram_mean.shape[0],
            sr=self.analysis_sr,
            hop_length=self.hop_length,
        )
        # tempo_frequencies includes lag=0 as the first bin, which maps to inf BPM.
        # Exclude that non-physical bin from peak picking.
        peak_tempo_bins = tempo_bins[1:] if tempo_bins.shape[0] > 1 else np.array([])
        peak_strength = tempogram_mean[1:] if tempogram_mean.shape[0] > 1 else np.array([])
        peaks = self._top_peaks(peak_tempo_bins, peak_strength, self.tempogram_peaks)
        profile_ds = self._downsample_curve(tempogram_mean, n=40) if tempogram_mean.size else []

        if phase is None:
            phase = {
                "pattern": [],
                "fft_mag": [],
                "bar_to_bar_var": 0.0,
                "n_bars_used": 0,
                "tempo_scale_used": 1.0,
                "bar_offset_used": 0,
            }

        return {
            "onset_source": "hpss_percussive",
            "periodicity": {
                "tempogram_top_peaks": peaks,
                "tempogram_profile_ds": profile_ds,
            },
            "phase": {
                "phase_bins_per_bar": int(self.phase_bins_per_bar),
                "pattern": self._clean_vector(phase["pattern"]),
                "fft_mag": self._clean_vector(phase["fft_mag"]),
                "stability": {
                    "bar_to_bar_var": float(phase["bar_to_bar_var"]),
                    "n_bars_used": int(phase["n_bars_used"]),
                },
                "tempo_scale_used": float(phase["tempo_scale_used"]),
                "bar_offset_used": int(phase["bar_offset_used"]),
            },
        }

    def _resample_onset_pattern(
        self,
        *,
        onset_env: Any,
        onset_times: Any,
        start_s: float,
        end_s: float,
        bins: int,
    ) -> np.ndarray | None:
        if end_s <= start_s:
            return None
        mask = (onset_times >= start_s) & (onset_times < end_s)
        if not np.any(mask):
            return None
        times = onset_times[mask]
        values = onset_env[mask]
        if len(times) < 2:
            return None
        t_norm = (times - start_s) / (end_s - start_s)
        grid = np.linspace(0.0, 1.0, bins)
        pattern = np.interp(grid, t_norm, values)
        return pattern

    def _mean_at_times(self, values: Any, times: Any, sample_times: list[float]) -> float:
        if values is None or len(values) == 0 or not sample_times:
            return 0.0
        sample_times_arr = np.asarray(sample_times)
        idx = np.searchsorted(times, sample_times_arr)
        idx = np.clip(idx, 0, len(values) - 1)
        return float(np.mean(values[idx]))

    def _top_peaks(self, bpm: Any, strength: Any, k: int) -> list[dict[str, float]]:
        if strength is None or len(strength) == 0:
            return []
        bpm_arr = np.asarray(bpm, dtype=float).reshape(-1)
        strength_arr = np.asarray(strength, dtype=float).reshape(-1)

        n = min(bpm_arr.shape[0], strength_arr.shape[0])
        if n == 0:
            return []
        bpm_arr = bpm_arr[:n]
        strength_arr = strength_arr[:n]

        valid = (
            np.isfinite(bpm_arr)
            & np.isfinite(strength_arr)
            & (bpm_arr > 0.0)
            & (strength_arr > 0.0)
        )
        if not np.any(valid):
            return []

        bpm_valid = bpm_arr[valid]
        strength_valid = strength_arr[valid]

        # Prefer musically plausible tempo peaks; fall back to all finite bins if needed.
        plausible = (bpm_valid >= 30.0) & (bpm_valid <= 300.0)
        if np.any(plausible):
            bpm_valid = bpm_valid[plausible]
            strength_valid = strength_valid[plausible]

        indices = np.argsort(strength_valid)[::-1]
        peaks: list[dict[str, float]] = []
        for idx in indices[:k]:
            peaks.append({"bpm": float(bpm_valid[idx]), "strength": float(strength_valid[idx])})
        return peaks

    def _scale_beats(self, beats_s: list[float], tempo_scale: float) -> list[float]:
        if not beats_s:
            return []
        if tempo_scale == 1.0:
            return beats_s
        if tempo_scale == 0.5:
            return beats_s[::2] if len(beats_s) > 1 else beats_s
        if tempo_scale == 2.0:
            scaled: list[float] = []
            for a, b in zip(beats_s[:-1], beats_s[1:]):
                mid = (a + b) * 0.5
                scaled.extend([a, mid])
            scaled.append(beats_s[-1])
            return scaled
        return beats_s

    def _build_beat_positions(self, n_beats: int, beats_per_bar: int, offset: int) -> list[int]:
        return [((i - offset) % beats_per_bar) + 1 for i in range(n_beats)]

    def _tempo_from_beats(self, beats_s: list[float]) -> tuple[float | None, dict[str, float]]:
        if len(beats_s) < 2:
            return None, {}
        intervals = np.diff(np.asarray(beats_s))
        intervals = intervals[intervals > 0]
        if len(intervals) == 0:
            return None, {}
        bpm_inst = 60.0 / intervals
        bpm = float(np.median(bpm_inst))
        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))
        cv = float(std_interval / mean_interval) if mean_interval > 0 else 0.0
        bpm_std = float(np.std(bpm_inst))
        return bpm, {
            "mean_s": mean_interval,
            "std_s": std_interval,
            "cv": cv,
            "bpm_std": bpm_std,
        }

    def _tempo_candidates(self, bpm: float | None) -> list[float]:
        if bpm is None:
            return []
        candidates = [bpm, bpm * 0.5, bpm * 2.0]
        unique = []
        for c in candidates:
            if c <= 0:
                continue
            if all(abs(c - u) > 0.1 for u in unique):
                unique.append(c)
        return [float(round(c, 4)) for c in unique]

    def _harmony_features(self, *, y: Any, sr: int) -> dict[str, Any]:
        if len(y) == 0:
            return {}
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        key = self._estimate_key(chroma_mean)
        return {
            "chroma_mean": self._clean_vector(chroma_mean.tolist()),
            "chroma_std": self._clean_vector(chroma_std.tolist()),
            "key": key,
            "source": {"feature": "chroma_cqt", "hop_length": self.hop_length},
        }

    def _estimate_key(self, chroma_mean: Any) -> dict[str, Any]:
        chroma = np.asarray(chroma_mean, dtype=float)
        if chroma.size != 12 or np.sum(chroma) == 0:
            return {"tonic": None, "mode": None, "confidence": 0.0}

        chroma = chroma / (np.sum(chroma) + 1e-9)
        major_profile = np.asarray(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
            dtype=float,
        )
        minor_profile = np.asarray(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
            dtype=float,
        )
        major_profile /= np.sum(major_profile)
        minor_profile /= np.sum(minor_profile)

        scores: list[tuple[str, float]] = []
        pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for i in range(12):
            corr_major = float(np.corrcoef(chroma, np.roll(major_profile, i))[0, 1])
            corr_minor = float(np.corrcoef(chroma, np.roll(minor_profile, i))[0, 1])
            scores.append((f"{pitch_classes[i]}:major", corr_major))
            scores.append((f"{pitch_classes[i]}:minor", corr_minor))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_label, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else -1.0
        tonic, mode = best_label.split(":")
        confidence = (best_score - second_score) / (abs(best_score) + 1e-9)
        confidence = max(0.0, min(1.0, confidence))
        return {"tonic": tonic, "mode": mode, "confidence": float(confidence)}

    def _timbre_features(self, *, y: Any, sr: int) -> dict[str, Any]:
        if len(y) == 0:
            return {}
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        return {
            "mfcc_mean": self._clean_vector(mfcc_mean.tolist()),
            "mfcc_std": self._clean_vector(mfcc_std.tolist()),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "spectral_flatness_std": float(np.std(flatness)),
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "zero_crossing_rate_std": float(np.std(zcr)),
        }

    def _spectral_features(self, *, y: Any, sr: int) -> dict[str, Any]:
        if len(y) == 0:
            return {}
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)
        return {
            "centroid_mean": float(np.mean(centroid)),
            "centroid_std": float(np.std(centroid)),
            "bandwidth_mean": float(np.mean(bandwidth)),
            "bandwidth_std": float(np.std(bandwidth)),
            "rolloff_mean": float(np.mean(rolloff)),
            "rolloff_std": float(np.std(rolloff)),
        }

    def _rms_curve(self, y: Any, sr: int) -> tuple[np.ndarray, np.ndarray]:
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=sr,
            hop_length=self.hop_length,
        )
        return rms, times

    def _spectral_flux(self, y: Any, sr: int) -> tuple[np.ndarray, np.ndarray]:
        if len(y) == 0:
            return np.array([]), np.array([])
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=self.hop_length))
        if S.shape[1] < 2:
            return np.array([]), np.array([])
        flux = np.maximum(0.0, np.diff(S, axis=1)).mean(axis=0)
        times = librosa.frames_to_time(
            np.arange(1, S.shape[1]),
            sr=sr,
            hop_length=self.hop_length,
        )
        return flux, times

    def _chroma_flux(self, y: Any, sr: int) -> tuple[np.ndarray, np.ndarray]:
        if len(y) == 0:
            return np.array([]), np.array([])
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        if chroma.shape[1] < 2:
            return np.array([]), np.array([])
        flux = np.mean(np.abs(np.diff(chroma, axis=1)), axis=0)
        times = librosa.frames_to_time(
            np.arange(1, chroma.shape[1]),
            sr=sr,
            hop_length=self.hop_length,
        )
        return flux, times

    def _mean_in_window(self, values: Any, times: Any, start_s: float, end_s: float) -> float:
        if values is None or len(values) == 0:
            return 0.0
        mask = (times >= start_s) & (times < end_s)
        if not np.any(mask):
            return 0.0
        return float(np.mean(values[mask]))

    def _slope_in_window(self, values: Any, times: Any, start_s: float, end_s: float) -> float:
        if values is None or len(values) < 2:
            return 0.0
        mask = (times >= start_s) & (times < end_s)
        if np.sum(mask) < 2:
            return 0.0
        x = times[mask]
        y = values[mask]
        if np.allclose(x, x[0]):
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def _downsample_curve(self, curve: Any, n: int = 120) -> list[float]:
        arr = np.asarray(curve, dtype=float)
        if arr.size == 0:
            return []
        if arr.size <= n:
            return self._clean_vector(arr.tolist())
        xs = np.linspace(0, arr.size - 1, n)
        ys = np.interp(xs, np.arange(arr.size), arr)
        return self._clean_vector(ys.tolist())

    def _normalize_pattern(self, pattern: np.ndarray) -> list[float]:
        if pattern.size == 0:
            return []
        max_val = float(np.max(pattern))
        if max_val <= 0:
            return self._clean_vector(pattern.tolist())
        return self._clean_vector((pattern / max_val).tolist())

    def _fft_mag(self, pattern: list[float], n: int = 16) -> list[float]:
        if not pattern:
            return []
        arr = np.asarray(pattern, dtype=float)
        fft = np.fft.rfft(arr)
        mags = np.abs(fft)[1 : n + 1]
        return self._clean_vector(mags.tolist())

    def _clean_floats(self, values: list[float]) -> list[float]:
        out: list[float] = []
        for v in values:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                out.append(f)
        return out

    def _clean_vector(self, values: list[float]) -> list[float]:
        out: list[float] = []
        for v in values:
            try:
                f = float(v)
            except (TypeError, ValueError):
                f = 0.0
            if not math.isfinite(f):
                f = 0.0
            out.append(f)
        return out


class PlaceholderMixAnalyzer:
    """
    Placeholder analyzer:
    - reads no actual audio
    - returns minimal analysis_json stub
    """

    async def analyze(self, abs_path: str, metadata_json: dict) -> AnalysisResult:
        analysis = {
            "schema_version": 2,
            "track": {"duration_s": 0.0, "sr": 0, "hop_length": 0, "overview": {}},
            "sections": {
                "switch_in": {
                    "start_s": 0.0,
                    "end_s": 0.0,
                    "quality": {},
                    "rhythm": {},
                    "harmony": {},
                    "timbre": {},
                    "spectral": {},
                    "energy": {},
                },
                "switch_out": {
                    "start_s": 0.0,
                    "end_s": 0.0,
                    "quality": {},
                    "rhythm": {},
                    "harmony": {},
                    "timbre": {},
                    "spectral": {},
                    "energy": {},
                },
            },
            "debug": {"errors": ["placeholder"]},
        }

        return AnalysisResult(
            metadata_json=metadata_json,
            analysis_json=analysis,
        )


class MixAnalyzerGlobalHPSS(MixAnalyzer):
    """
    Variant that performs full-track HPSS and uses CV-based local/global beat source selection.
    """

    USE_GLOBAL_HPSS = True
    PREFER_LOCAL_BEATS = False
    VARIANT_NAME = "global_hpss_cv_beats"
