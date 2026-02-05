from __future__ import annotations

import asyncio
import io
import importlib.util
import logging
import math
import struct
import time
import wave
from dataclasses import dataclass
from typing import Any, Protocol, cast
import uuid

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter


@dataclass(frozen=True)
class TrackInput:
    mix_item_id: uuid.UUID
    abs_path: str
    metadata_json: dict
    analysis_json: dict | None


@dataclass(frozen=True)
class SegmentSpec:
    position: int
    mix_item_id: uuid.UUID
    start_ms: int
    end_ms: int
    source_start_ms: int
    song_name: str
    artist_name: str


@dataclass(frozen=True)
class RenderResult:
    audio_bytes: bytes
    mime: str
    length_ms: int
    segments: list[SegmentSpec]
    debug: dict[str, Any] | None = None


class MixRenderer(Protocol):
    async def render(self, tracks: list[TrackInput], fixed_tracks: list[TrackInput]) -> RenderResult:
        ...


@dataclass(frozen=True)
class _KeyInfo:
    tonic: str | None
    mode: str | None
    confidence: float


@dataclass(frozen=True)
class _SectionInfo:
    start_s: float
    end_s: float
    bpm_local: float | None
    bpm_candidates: tuple[float, ...]
    beats_s: tuple[float, ...]
    downbeats_s: tuple[float, ...]
    beats_per_bar: int
    meter_confidence: float
    tempo_scale_used: float
    phase_pattern: tuple[float, ...]
    phase_var: float
    tempogram_peaks: tuple[float, ...]
    key: _KeyInfo
    mfcc_mean: tuple[float, ...]
    mfcc_std: tuple[float, ...]
    onset_clarity: float
    tempo_stability: float
    spectral_flux: float
    chroma_flux: float
    energy_ramp: float


@dataclass(frozen=True)
class _TrackFeatures:
    track: TrackInput
    switch_in: _SectionInfo | None
    switch_out: _SectionInfo | None
    valid: bool


@dataclass(frozen=True)
class _TransitionOption:
    incoming_bpm: float
    target_bpm: float
    delta_bpm_pct: float
    stretch_rate: float
    key_penalty: float
    signature_penalty: float
    instability_penalty: float
    secondary_cost: float
    total_cost: float
    key_clash: bool


@dataclass(frozen=True)
class _TransitionPlan:
    from_id: uuid.UUID
    to_id: uuid.UUID
    option: _TransitionOption
    overlap_bars: int
    overlap_s: float


@dataclass
class _PlacedTrack:
    track: TrackInput
    source_start_s: float
    stretch_rate: float
    mix_start_s: float


class DeterministicMixRenderer:
    RENDER_SR = 44100
    CHANNELS = 2
    SAMPWIDTH = 2

    SIMPLE_OVERLAP_S = 6.0
    MIN_OVERLAP_S = 4.0
    MAX_OVERLAP_S = 20.0
    MAX_DP_TRACKS = 10
    MAX_STRETCH_DELTA_PCT = 6.0
    MIN_STRETCH_RATE = 0.94
    MAX_STRETCH_RATE = 1.06
    BASS_SWAP_HPF_HZ = 180.0
    # WAV is deterministic and fast to encode (use MP3 only when you explicitly need smaller files).
    DEFAULT_OUTPUT_MIME = "audio/wav"
    TRIM_PAD_PRE_S = 0.25
    TRIM_PAD_POST_S = 0.75
    MIN_TRIMMED_WINDOW_S = 8.0

    def __init__(
        self,
        *,
        enable_timing_logs: bool = False,
        enable_debug_logs: bool = False,
        output_mime: str | None = None,
        resample_res_type: str | None = None,
        time_stretch_n_fft: int = 1024,
        time_stretch_hop_length: int = 256,
    ) -> None:
        self.enable_timing_logs = enable_timing_logs
        self.enable_debug_logs = enable_debug_logs
        self.output_mime = output_mime or self.DEFAULT_OUTPUT_MIME

        # Prefer SoXR (C-accelerated) when available; fallback to resampy (kaiser_fast).
        if resample_res_type:
            self.resample_res_type = str(resample_res_type)
        else:
            self.resample_res_type = (
                "soxr_hq" if importlib.util.find_spec("soxr") is not None else "kaiser_fast"
            )

        # Librosa time-stretch defaults (2048/512) are noticeably slower for long audio; use smaller FFTs by default.
        self.time_stretch_n_fft = max(256, int(time_stretch_n_fft))
        self.time_stretch_hop_length = max(64, int(time_stretch_hop_length))
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def render(self, tracks: list[TrackInput], fixed_tracks: list[TrackInput]) -> RenderResult:
        return await asyncio.to_thread(self._render_sync, tracks, fixed_tracks)

    def _render_sync(self, tracks: list[TrackInput], fixed_tracks: list[TrackInput]) -> RenderResult:
        total_start = time.perf_counter()
        debug: dict[str, Any] = {
            "timing_s": {},
            "decisions": {
                "input_track_ids": [str(t.mix_item_id) for t in tracks],
                "requested_prefix_ids": [str(t.mix_item_id) for t in fixed_tracks],
            },
            "fallbacks": [],
            "errors": [],
        }

        step_start = time.perf_counter()
        unique_tracks, prefix_ids = self._canonicalize_tracks(tracks, fixed_tracks)
        self._record_timing(debug, "canonicalize_tracks", step_start)
        debug["decisions"]["unique_track_ids"] = [str(t.mix_item_id) for t in unique_tracks]
        debug["decisions"]["applied_prefix_ids"] = [str(tid) for tid in prefix_ids]

        if not unique_tracks:
            silence = np.zeros((self.RENDER_SR, self.CHANNELS), dtype=np.float32)
            step_start = time.perf_counter()
            audio_bytes = self._encode_output_bytes(silence)
            self._record_timing(debug, "encode_output", step_start)
            debug["decisions"]["render_mode"] = "empty_input_silence"
            self._record_timing(debug, "total", total_start)
            self._emit_render_debug(debug)
            return RenderResult(
                audio_bytes=audio_bytes,
                mime=self.output_mime,
                length_ms=1000,
                segments=[],
                debug=debug,
            )

        step_start = time.perf_counter()
        features = {t.mix_item_id: self._extract_track_features(t) for t in unique_tracks}
        self._record_timing(debug, "extract_features", step_start)
        all_valid = all(features[t.mix_item_id].valid for t in unique_tracks)
        debug["decisions"]["all_analysis_valid"] = all_valid
        invalid_ids = [str(t.mix_item_id) for t in unique_tracks if not features[t.mix_item_id].valid]
        if invalid_ids:
            debug["decisions"]["invalid_analysis_track_ids"] = invalid_ids
            debug["fallbacks"].append("append_order_due_to_invalid_analysis")

        step_start = time.perf_counter()
        if all_valid:
            ordered_tracks, ordering_strategy = self._order_tracks(unique_tracks, prefix_ids, features)
        else:
            ordered_tracks = self._append_order(unique_tracks, prefix_ids)
            ordering_strategy = "append_order"
        self._record_timing(debug, "order_tracks", step_start)
        debug["decisions"]["ordering_strategy"] = ordering_strategy
        debug["decisions"]["final_order_ids"] = [str(t.mix_item_id) for t in ordered_tracks]

        plans: list[_TransitionPlan] = []
        if len(ordered_tracks) == 1:
            debug["decisions"]["render_mode"] = "single_track"
            step_start = time.perf_counter()
            mix_audio, segments = self._render_single_track(ordered_tracks[0], features.get(ordered_tracks[0].mix_item_id))
            self._record_timing(debug, "render_audio", step_start)
        elif all_valid:
            debug["decisions"]["render_mode"] = "planned_transitions"
            step_start = time.perf_counter()
            plans = self._build_transition_plan(ordered_tracks, features)
            self._record_timing(debug, "build_transition_plan", step_start)
            debug["decisions"]["transitions"] = self._plan_debug_payload(plans)
            step_start = time.perf_counter()
            mix_audio, segments = self._render_with_plan(ordered_tracks, features, plans)
            self._record_timing(debug, "render_audio", step_start)
        else:
            debug["decisions"]["render_mode"] = "simple_crossfades"
            step_start = time.perf_counter()
            mix_audio, segments = self._render_simple_crossfades(ordered_tracks)
            self._record_timing(debug, "render_audio", step_start)
            debug["decisions"]["simple_overlap_s"] = self.SIMPLE_OVERLAP_S

        step_start = time.perf_counter()
        audio_bytes = self._encode_output_bytes(mix_audio)
        self._record_timing(debug, "encode_output", step_start)
        length_ms = int(round((len(mix_audio) / self.RENDER_SR) * 1000))
        debug["decisions"]["length_ms"] = length_ms
        debug["decisions"]["segments_count"] = len(segments)
        debug["decisions"]["encoded_bytes"] = len(audio_bytes)
        self._record_timing(debug, "total", total_start)
        self._emit_render_debug(debug)
        return RenderResult(
            audio_bytes=audio_bytes,
            mime=self.output_mime,
            length_ms=length_ms,
            segments=segments,
            debug=debug,
        )

    # ---------- Validation / Ordering ----------

    def _canonicalize_tracks(
        self, tracks: list[TrackInput], fixed_tracks: list[TrackInput]
    ) -> tuple[list[TrackInput], list[uuid.UUID]]:
        unique: dict[uuid.UUID, TrackInput] = {}
        ordered_ids: list[uuid.UUID] = []
        for tr in tracks:
            if tr.mix_item_id not in unique:
                unique[tr.mix_item_id] = tr
                ordered_ids.append(tr.mix_item_id)

        fixed_ids: list[uuid.UUID] = []
        for tr in fixed_tracks:
            tid = tr.mix_item_id
            if tid in unique and tid not in fixed_ids:
                fixed_ids.append(tid)

        return [unique[tid] for tid in ordered_ids], fixed_ids

    def _append_order(self, tracks: list[TrackInput], prefix_ids: list[uuid.UUID]) -> list[TrackInput]:
        by_id = {t.mix_item_id: t for t in tracks}
        prefix = [by_id[tid] for tid in prefix_ids]
        suffix = [t for t in tracks if t.mix_item_id not in set(prefix_ids)]
        return prefix + suffix

    def _order_tracks(
        self,
        tracks: list[TrackInput],
        prefix_ids: list[uuid.UUID],
        features: dict[uuid.UUID, _TrackFeatures],
    ) -> tuple[list[TrackInput], str]:
        by_id = {t.mix_item_id: t for t in tracks}
        prefix = [by_id[tid] for tid in prefix_ids]
        remaining = [t for t in tracks if t.mix_item_id not in set(prefix_ids)]
        if len(remaining) <= 1:
            return prefix + remaining, "append_order"

        cost_map: dict[tuple[uuid.UUID, uuid.UUID], float] = {}
        for a in tracks:
            for b in tracks:
                if a.mix_item_id == b.mix_item_id:
                    continue
                opt = self._best_transition_option(features[a.mix_item_id], features[b.mix_item_id])
                cost_map[(a.mix_item_id, b.mix_item_id)] = opt.total_cost

        anchor = prefix[-1].mix_item_id if prefix else None
        remaining_ids = [t.mix_item_id for t in remaining]
        if len(remaining_ids) <= self.MAX_DP_TRACKS:
            order_ids = self._optimal_path_ids(remaining_ids, anchor, cost_map)
            strategy = "held_karp_dp"
        else:
            order_ids = self._greedy_path_ids(remaining_ids, anchor, cost_map)
            strategy = "greedy"
        return prefix + [by_id[i] for i in order_ids], strategy

    def _optimal_path_ids(
        self,
        remaining_ids: list[uuid.UUID],
        anchor_id: uuid.UUID | None,
        cost_map: dict[tuple[uuid.UUID, uuid.UUID], float],
    ) -> list[uuid.UUID]:
        n = len(remaining_ids)
        ids = remaining_ids
        dp: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}

        for i in range(n):
            c0 = cost_map[(anchor_id, ids[i])] if anchor_id is not None else 0.0
            dp[(1 << i, i)] = (c0, (i,))

        for mask in range(1, 1 << n):
            for last in range(n):
                key = (mask, last)
                if key not in dp:
                    continue
                base_cost, base_path = dp[key]
                for nxt in range(n):
                    bit = 1 << nxt
                    if mask & bit:
                        continue
                    nm = mask | bit
                    nc = base_cost + cost_map[(ids[last], ids[nxt])]
                    npth = base_path + (nxt,)
                    cur = dp.get((nm, nxt))
                    if cur is None or self._cost_path_key(nc, npth, ids) < self._cost_path_key(cur[0], cur[1], ids):
                        dp[(nm, nxt)] = (nc, npth)

        full = (1 << n) - 1
        best: tuple[float, tuple[int, ...]] | None = None
        for last in range(n):
            cur = dp.get((full, last))
            if cur is None:
                continue
            if best is None or self._cost_path_key(cur[0], cur[1], ids) < self._cost_path_key(best[0], best[1], ids):
                best = cur
        if best is None:
            return sorted(remaining_ids, key=lambda x: str(x))
        return [ids[i] for i in best[1]]

    def _greedy_path_ids(
        self,
        remaining_ids: list[uuid.UUID],
        anchor_id: uuid.UUID | None,
        cost_map: dict[tuple[uuid.UUID, uuid.UUID], float],
    ) -> list[uuid.UUID]:
        todo = set(remaining_ids)
        path: list[uuid.UUID] = []
        if anchor_id is None:
            start = min(
                todo,
                key=lambda tid: (
                    sum(cost_map[(tid, other)] for other in todo if other != tid),
                    str(tid),
                ),
            )
            path.append(start)
            todo.remove(start)

        cur = anchor_id if anchor_id is not None else path[-1]
        while todo:
            nxt = min(todo, key=lambda tid: (cost_map[(cur, tid)], str(tid)))
            path.append(nxt)
            todo.remove(nxt)
            cur = nxt
        return path

    def _cost_path_key(self, cost: float, path: tuple[int, ...], ids: list[uuid.UUID]) -> tuple[float, tuple[str, ...]]:
        return (float(cost), tuple(str(ids[i]) for i in path))

    # ---------- Feature extraction ----------

    def _extract_track_features(self, track: TrackInput) -> _TrackFeatures:
        analysis: dict[str, Any] = track.analysis_json if isinstance(track.analysis_json, dict) else {}
        sections_obj = analysis.get("sections")
        if not isinstance(sections_obj, dict):
            return _TrackFeatures(track=track, switch_in=None, switch_out=None, valid=False)
        sections: dict[str, Any] = sections_obj

        switch_in = self._extract_section_info(sections.get("switch_in"))
        switch_out = self._extract_section_info(sections.get("switch_out"))
        valid = switch_in is not None and switch_out is not None and switch_in.bpm_local is not None and switch_out.bpm_local is not None
        return _TrackFeatures(track=track, switch_in=switch_in, switch_out=switch_out, valid=valid)

    def _extract_section_info(self, section: Any) -> _SectionInfo | None:
        section_d = self._as_dict(section)
        if not section_d:
            return None
        rhythm_d = self._as_dict(section_d.get("rhythm"))
        if not rhythm_d:
            return None

        bpm_local = self._to_optional_float(rhythm_d.get("bpm_local"))
        bpm_candidates = self._float_tuple(rhythm_d.get("bpm_candidates"))
        beats = self._float_tuple(rhythm_d.get("beats_s"))
        downbeats = self._float_tuple(rhythm_d.get("downbeats_s"))

        meter = self._as_dict(rhythm_d.get("meter"))
        beats_per_bar = self._to_int(meter.get("beats_per_bar"), default=4)
        meter_confidence = self._to_float(meter.get("confidence"), default=0.0)

        signature = self._as_dict(rhythm_d.get("signature"))
        phase = self._as_dict(signature.get("phase"))
        periodicity = self._as_dict(signature.get("periodicity"))

        phase_pattern = self._float_tuple(phase.get("pattern"))
        phase_stability = self._as_dict(phase.get("stability"))
        phase_var = self._to_float(phase_stability.get("bar_to_bar_var"), default=0.0)
        tempo_scale_used = self._to_float(phase.get("tempo_scale_used"), default=1.0)

        peaks: list[float] = []
        peaks_raw = periodicity.get("tempogram_top_peaks")
        peaks_list = peaks_raw if isinstance(peaks_raw, list) else []
        for p in peaks_list:
            if isinstance(p, dict):
                bpm = self._to_optional_float(p.get("bpm"))
                if bpm is not None and bpm > 0:
                    peaks.append(bpm)

        harmony = self._as_dict(section_d.get("harmony"))
        key_raw = self._as_dict(harmony.get("key"))
        key = _KeyInfo(
            tonic=self._to_str(key_raw.get("tonic")),
            mode=self._to_str(key_raw.get("mode")),
            confidence=self._to_float(key_raw.get("confidence"), default=0.0),
        )

        quality = self._as_dict(section_d.get("quality"))
        comp = self._as_dict(quality.get("components"))
        timbre = self._as_dict(section_d.get("timbre"))

        return _SectionInfo(
            start_s=self._to_float(section_d.get("start_s"), default=0.0),
            end_s=self._to_float(section_d.get("end_s"), default=0.0),
            bpm_local=bpm_local,
            bpm_candidates=bpm_candidates,
            beats_s=beats,
            downbeats_s=downbeats,
            beats_per_bar=max(1, beats_per_bar),
            meter_confidence=meter_confidence,
            tempo_scale_used=tempo_scale_used,
            phase_pattern=phase_pattern,
            phase_var=max(0.0, float(phase_var)),
            tempogram_peaks=tuple(peaks),
            key=key,
            mfcc_mean=self._float_tuple(timbre.get("mfcc_mean")),
            mfcc_std=self._float_tuple(timbre.get("mfcc_std")),
            onset_clarity=self._to_float(comp.get("onset_clarity"), default=0.0),
            tempo_stability=self._to_float(comp.get("tempo_stability"), default=0.0),
            spectral_flux=self._to_float(comp.get("spectral_flux"), default=0.0),
            chroma_flux=self._to_float(comp.get("chroma_flux"), default=0.0),
            energy_ramp=self._to_float(comp.get("energy_ramp"), default=0.0),
        )

    # ---------- Transition scoring / planning ----------

    def _build_transition_plan(
        self, ordered_tracks: list[TrackInput], features: dict[uuid.UUID, _TrackFeatures]
    ) -> list[_TransitionPlan]:
        """
        Build a sequential, tempo-consistent transition plan.

        Important: the renderer currently time-stretches the *entire rendered chunk* of each incoming track.
        That means a track's "effective BPM in the mix" is modified and persists until the next transition.

        To avoid overlaps where the outgoing audio is at a different (already-stretched) tempo than the
        incoming audio, we plan transitions against a running "mix BPM" that propagates forward.
        """
        plans: list[_TransitionPlan] = []

        if len(ordered_tracks) < 2:
            return plans

        first_feat = features.get(ordered_tracks[0].mix_item_id)
        first_out = first_feat.switch_out if first_feat is not None else None
        first_in = first_feat.switch_in if first_feat is not None else None
        current_bpm = (
            float(first_out.bpm_local)
            if first_out is not None and first_out.bpm_local is not None and first_out.bpm_local > 0
            else float(first_in.bpm_local)
            if first_in is not None and first_in.bpm_local is not None and first_in.bpm_local > 0
            else 120.0
        )

        for i in range(len(ordered_tracks) - 1):
            a = ordered_tracks[i]
            b = ordered_tracks[i + 1]
            option = self._best_transition_option(features[a.mix_item_id], features[b.mix_item_id], target_bpm=current_bpm)

            out_sec = features[a.mix_item_id].switch_out
            beats_per_bar = out_sec.beats_per_bar if out_sec is not None else 4
            bpm = option.target_bpm if option.target_bpm > 0 else 120.0
            bar_s = (60.0 / bpm) * beats_per_bar

            overlap_bars = 8
            if option.delta_bpm_pct > 4.0 or option.key_clash or option.signature_penalty > 0.55:
                overlap_bars = 4
            elif option.delta_bpm_pct < 1.5 and option.signature_penalty < 0.25 and not option.key_clash:
                overlap_bars = 12

            overlap_s = max(self.MIN_OVERLAP_S, min(self.MAX_OVERLAP_S, bar_s * overlap_bars))
            plans.append(
                _TransitionPlan(
                    from_id=a.mix_item_id,
                    to_id=b.mix_item_id,
                    option=option,
                    overlap_bars=overlap_bars,
                    overlap_s=overlap_s,
                )
            )

            # Propagate the actually-achieved tempo to keep subsequent transitions coherent.
            # (When clamped, option.target_bpm may not be fully achieved.)
            achieved = float(option.incoming_bpm * option.stretch_rate) if option.incoming_bpm > 0 else float(bpm)
            if math.isfinite(achieved) and achieved > 0:
                current_bpm = achieved
        return plans

    def _best_transition_option(
        self, a: _TrackFeatures, b: _TrackFeatures, *, target_bpm: float | None = None
    ) -> _TransitionOption:
        a_out = a.switch_out
        b_in = b.switch_in
        if b_in is None:
            return _TransitionOption(
                incoming_bpm=b_in.bpm_local if b_in and b_in.bpm_local else 120.0,
                target_bpm=float(target_bpm) if target_bpm is not None else (a_out.bpm_local if a_out and a_out.bpm_local else 120.0),
                delta_bpm_pct=99.0,
                stretch_rate=1.0,
                key_penalty=0.5,
                signature_penalty=0.5,
                instability_penalty=0.0,
                secondary_cost=2.0,
                total_cost=9999.0,
                key_clash=False,
            )
        if a_out is None or a_out.bpm_local is None:
            # Without an outgoing analysis section, we can still evaluate incoming candidates against
            # a provided target mix tempo (if any); otherwise use the incoming local tempo.
            fallback_target = float(target_bpm) if target_bpm is not None else (b_in.bpm_local if b_in.bpm_local else 120.0)
            return _TransitionOption(
                incoming_bpm=b_in.bpm_local if b_in.bpm_local else fallback_target,
                target_bpm=fallback_target,
                delta_bpm_pct=99.0,
                stretch_rate=1.0,
                key_penalty=0.5,
                signature_penalty=0.5,
                instability_penalty=0.0,
                secondary_cost=2.0,
                total_cost=9999.0,
                key_clash=False,
            )

        incoming = list(b_in.bpm_candidates)
        if b_in.bpm_local is not None:
            incoming.append(b_in.bpm_local)
        incoming = sorted({round(v, 6) for v in incoming if v > 0})
        if not incoming:
            incoming = [a_out.bpm_local]

        target = float(target_bpm) if target_bpm is not None and target_bpm > 0 else float(a_out.bpm_local)

        options: list[_TransitionOption] = []
        for bpm_in in incoming:
            raw_rate = target / bpm_in if bpm_in > 0 else 1.0
            stretch = float(np.clip(raw_rate, self.MIN_STRETCH_RATE, self.MAX_STRETCH_RATE))
            effective = bpm_in * stretch
            delta_pct = abs(effective - target) / max(target, 1e-9) * 100.0

            tempo_cost = delta_pct + max(0.0, delta_pct - self.MAX_STRETCH_DELTA_PCT) * 40.0
            key_pen = self._key_penalty(a_out.key, b_in.key)
            key_weight = min(max(a_out.key.confidence, 0.0), max(b_in.key.confidence, 0.0))
            key_pen_w = key_pen * key_weight
            sig_pen = self._signature_penalty(a_out, b_in)
            timbre_pen = self._timbre_penalty(a_out, b_in)
            sec = 0.8 * key_pen_w + 1.0 * sig_pen + 1.0 * timbre_pen
            total = tempo_cost * 100.0 + sec * 10.0

            options.append(
                _TransitionOption(
                    incoming_bpm=bpm_in,
                    target_bpm=target,
                    delta_bpm_pct=delta_pct,
                    stretch_rate=stretch,
                    key_penalty=key_pen_w,
                    signature_penalty=sig_pen,
                    instability_penalty=0.0,
                    secondary_cost=sec,
                    total_cost=total,
                    key_clash=(key_pen_w >= 0.55),
                )
            )

        options.sort(
            key=lambda o: (
                o.total_cost,
                o.delta_bpm_pct,
                o.secondary_cost,
                o.incoming_bpm,
                str(b.track.mix_item_id),
            )
        )
        return options[0]

    def _key_penalty(self, a: _KeyInfo, b: _KeyInfo) -> float:
        if not a.tonic or not b.tonic:
            return 0.3
        ai = self._fifths_index(a.tonic)
        bi = self._fifths_index(b.tonic)
        if ai is None or bi is None:
            return 0.3
        dist = min(abs(ai - bi), 12 - abs(ai - bi))
        mode_pen = 0.2 if a.mode and b.mode and a.mode != b.mode else 0.0
        return float(min(1.0, dist / 6.0 + mode_pen))

    def _fifths_index(self, tonic: str) -> int | None:
        order = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
        aliases = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
        tonic = aliases.get(tonic, tonic)
        try:
            return order.index(tonic)
        except ValueError:
            return None

    def _signature_penalty(self, a: _SectionInfo, b: _SectionInfo) -> float:
        p_pen = 0.5
        if a.phase_pattern and b.phase_pattern:
            va = np.asarray(a.phase_pattern, dtype=np.float64)
            vb = np.asarray(b.phase_pattern, dtype=np.float64)
            n = min(len(va), len(vb))
            if n > 0:
                va = va[:n]
                vb = vb[:n]
                den = np.linalg.norm(va) * np.linalg.norm(vb)
                if den > 0:
                    cos = float(np.dot(va, vb) / den)
                    p_pen = float(np.clip(1.0 - cos, 0.0, 1.0))

        t_pen = 0.5
        if a.tempogram_peaks and b.tempogram_peaks:
            pa = np.asarray(a.tempogram_peaks[:5], dtype=np.float64)
            pb = np.asarray(b.tempogram_peaks[:5], dtype=np.float64)
            diffs = [float(np.min(np.abs(pb - x) / np.maximum(x, 1e-9))) for x in pa]
            if diffs:
                t_pen = float(np.clip(np.mean(diffs), 0.0, 1.0))
        return float(0.7 * p_pen + 0.3 * t_pen)

    def _timbre_penalty(self, a: _SectionInfo, b: _SectionInfo) -> float:
        n = min(len(a.mfcc_mean), len(b.mfcc_mean), len(a.mfcc_std), len(b.mfcc_std))
        if n <= 0:
            return 0.5

        mean_a = np.nan_to_num(np.asarray(a.mfcc_mean[:n], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        mean_b = np.nan_to_num(np.asarray(b.mfcc_mean[:n], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        std_a = np.nan_to_num(np.abs(np.asarray(a.mfcc_std[:n], dtype=np.float64)), nan=0.0, posinf=0.0, neginf=0.0)
        std_b = np.nan_to_num(np.abs(np.asarray(b.mfcc_std[:n], dtype=np.float64)), nan=0.0, posinf=0.0, neginf=0.0)

        pooled_var = np.maximum(0.5 * (np.square(std_a) + np.square(std_b)), 1e-6)
        diff = mean_a - mean_b
        md_sq = float(np.sum(np.square(diff) / pooled_var))
        if not math.isfinite(md_sq) or md_sq < 0.0:
            return 0.5

        dist = math.sqrt(md_sq)
        scale = max(math.sqrt(float(n)), 1.0)
        normalized = dist / (dist + scale)
        return float(np.clip(normalized, 0.0, 1.0))

    # ---------- Rendering ----------

    def _render_single_track(
        self,
        track: TrackInput,
        feat: _TrackFeatures | None = None,
    ) -> tuple[np.ndarray, list[SegmentSpec]]:
        # First/only track must begin at source start for predictable playback.
        region = None
        source_start_s = 0.0
        source_end_s: float | None = None
        if region is not None:
            source_start_s, source_end_s = region

        audio = (
            self._load_audio_stereo_region(track.abs_path, start_s=source_start_s, end_s=source_end_s)
            if source_end_s is not None
            else self._load_audio_stereo(track.abs_path)
        )
        length_ms = int(round((len(audio) / self.RENDER_SR) * 1000))
        meta = track.metadata_json or {}
        seg = SegmentSpec(
            position=0,
            mix_item_id=track.mix_item_id,
            start_ms=0,
            end_ms=length_ms,
            source_start_ms=int(round(source_start_s * 1000)),
            song_name=str(meta.get("song_name", "")),
            artist_name=str(meta.get("artist_name", "")),
        )
        return audio, [seg]

    def _render_with_plan(
        self,
        ordered_tracks: list[TrackInput],
        features: dict[uuid.UUID, _TrackFeatures],
        plans: list[_TransitionPlan],
    ) -> tuple[np.ndarray, list[SegmentSpec]]:
        first = ordered_tracks[0]
        # Keep first track anchored to source start (not switch-in).
        first_source_start_s = 0.0
        first_source_end_s: float | None = None
        first_audio = (
            self._load_audio_stereo_region(first.abs_path, start_s=first_source_start_s, end_s=first_source_end_s)
            if first_source_end_s is not None
            else self._load_audio_stereo(first.abs_path)
        )
        chunks: list[np.ndarray] = [first_audio]
        total_len_n = int(len(first_audio))

        first_meta = first.metadata_json or {}
        segments: list[SegmentSpec] = [
            SegmentSpec(
                position=0,
                mix_item_id=first.mix_item_id,
                start_ms=0,
                end_ms=int(round((total_len_n / self.RENDER_SR) * 1000)),
                source_start_ms=int(round(first_source_start_s * 1000)),
                song_name=str(first_meta.get("song_name", "")),
                artist_name=str(first_meta.get("artist_name", "")),
            )
        ]
        placed = _PlacedTrack(track=first, source_start_s=first_source_start_s, stretch_rate=1.0, mix_start_s=0.0)

        for i, plan in enumerate(plans):
            incoming_track = ordered_tracks[i + 1]
            incoming_feat = features[incoming_track.mix_item_id]
            outgoing_feat = features[placed.track.mix_item_id]

            # Load only the section we actually render: switch-in -> switch-out (with small pads).
            region = self._track_region(incoming_feat)
            region_start_s = region[0] if region is not None else 0.0
            region_end_s = region[1] if region is not None else None

            source_start_s = max(region_start_s, self._incoming_source_start(incoming_feat))
            incoming = (
                self._load_audio_stereo_region(incoming_track.abs_path, start_s=source_start_s, end_s=region_end_s)
                if region_end_s is not None
                else self._load_audio_stereo(incoming_track.abs_path)
            )
            if incoming.size == 0:
                # Conservative fallback: try the full track if region read failed.
                source_start_s = 0.0
                incoming = self._load_audio_stereo(incoming_track.abs_path)

            incoming = self._time_stretch_stereo(incoming, plan.option.stretch_rate)

            overlap_n = int(round(plan.overlap_s * self.RENDER_SR))

            # Operate on the last chunk only (avoid full-mix copies).
            out_chunk = chunks[-1]
            prefix_len_n = total_len_n - int(len(out_chunk))
            out_start_n = prefix_len_n

            # Ensure overlap is feasible within the outgoing chunk.
            overlap_n = int(np.clip(overlap_n, 1, max(1, len(out_chunk) - 1)))
            default_start_n = max(out_start_n, total_len_n - overlap_n)
            transition_start_n = self._pick_outgoing_transition_start(
                default_transition_start_n=default_start_n,
                mix_len_n=total_len_n,
                overlap_n=overlap_n,
                outgoing_feat=outgoing_feat,
                placed=placed,
            )
            transition_start_n = int(np.clip(transition_start_n, out_start_n, total_len_n - overlap_n))
            start_in_chunk = transition_start_n - out_start_n
            overlap_n = int(np.clip(overlap_n, 1, max(1, len(out_chunk) - start_in_chunk)))

            if len(incoming) < overlap_n:
                incoming = np.vstack([incoming, np.zeros((overlap_n - len(incoming), 2), dtype=np.float32)])

            out_ov = out_chunk[start_in_chunk : start_in_chunk + overlap_n]
            in_ov = self._apply_bass_swap(incoming[:overlap_n])
            blend = self._equal_power_blend(out_ov, in_ov)
            tail = incoming[overlap_n:]

            # Replace outgoing chunk tail (truncate anything after the overlap) and append incoming tail.
            new_out_chunk = np.vstack([out_chunk[:start_in_chunk], blend])
            chunks[-1] = new_out_chunk
            if tail.size:
                chunks.append(tail)

            total_len_n = prefix_len_n + int(len(new_out_chunk)) + int(len(tail))

            prev = segments[-1]
            segments[-1] = SegmentSpec(
                position=prev.position,
                mix_item_id=prev.mix_item_id,
                start_ms=prev.start_ms,
                end_ms=int(round(((transition_start_n + overlap_n) / self.RENDER_SR) * 1000)),
                source_start_ms=prev.source_start_ms,
                song_name=prev.song_name,
                artist_name=prev.artist_name,
            )

            meta = incoming_track.metadata_json or {}
            segments.append(
                SegmentSpec(
                    position=i + 1,
                    mix_item_id=incoming_track.mix_item_id,
                    start_ms=int(round((transition_start_n / self.RENDER_SR) * 1000)),
                    end_ms=int(round((total_len_n / self.RENDER_SR) * 1000)),
                    source_start_ms=int(round(source_start_s * 1000)),
                    song_name=str(meta.get("song_name", "")),
                    artist_name=str(meta.get("artist_name", "")),
                )
            )

            placed = _PlacedTrack(
                track=incoming_track,
                source_start_s=source_start_s,
                stretch_rate=plan.option.stretch_rate,
                mix_start_s=transition_start_n / self.RENDER_SR,
            )

        if segments:
            last = segments[-1]
            segments[-1] = SegmentSpec(
                position=last.position,
                mix_item_id=last.mix_item_id,
                start_ms=last.start_ms,
                end_ms=int(round((total_len_n / self.RENDER_SR) * 1000)),
                source_start_ms=last.source_start_ms,
                song_name=last.song_name,
                artist_name=last.artist_name,
            )
        mix_audio = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
        return mix_audio, segments

    def _render_simple_crossfades(self, ordered_tracks: list[TrackInput]) -> tuple[np.ndarray, list[SegmentSpec]]:
        first = ordered_tracks[0]
        first_audio = self._load_audio_stereo(first.abs_path)
        chunks: list[np.ndarray] = [first_audio]
        total_len_n = int(len(first_audio))
        meta = first.metadata_json or {}
        segments: list[SegmentSpec] = [
            SegmentSpec(
                position=0,
                mix_item_id=first.mix_item_id,
                start_ms=0,
                end_ms=int(round((total_len_n / self.RENDER_SR) * 1000)),
                source_start_ms=0,
                song_name=str(meta.get("song_name", "")),
                artist_name=str(meta.get("artist_name", "")),
            )
        ]

        ov_n = int(round(self.SIMPLE_OVERLAP_S * self.RENDER_SR))
        for i in range(1, len(ordered_tracks)):
            tr = ordered_tracks[i]
            audio = self._load_audio_stereo(tr.abs_path)
            out_chunk = chunks[-1]
            prefix_len_n = total_len_n - int(len(out_chunk))
            ov = int(np.clip(ov_n, 1, max(1, len(out_chunk) - 1)))
            if len(audio) < ov:
                audio = np.vstack([audio, np.zeros((ov - len(audio), 2), dtype=np.float32)])
            start_in_chunk = len(out_chunk) - ov
            start_n = prefix_len_n + start_in_chunk
            blend = self._equal_power_blend(out_chunk[start_in_chunk : start_in_chunk + ov], audio[:ov])

            new_out_chunk = np.vstack([out_chunk[:start_in_chunk], blend])
            chunks[-1] = new_out_chunk
            tail = audio[ov:]
            if tail.size:
                chunks.append(tail)
            total_len_n = prefix_len_n + int(len(new_out_chunk)) + int(len(tail))

            prev = segments[-1]
            segments[-1] = SegmentSpec(
                position=prev.position,
                mix_item_id=prev.mix_item_id,
                start_ms=prev.start_ms,
                end_ms=int(round(((start_n + ov) / self.RENDER_SR) * 1000)),
                source_start_ms=prev.source_start_ms,
                song_name=prev.song_name,
                artist_name=prev.artist_name,
            )
            meta = tr.metadata_json or {}
            segments.append(
                SegmentSpec(
                    position=i,
                    mix_item_id=tr.mix_item_id,
                    start_ms=int(round((start_n / self.RENDER_SR) * 1000)),
                    end_ms=int(round((total_len_n / self.RENDER_SR) * 1000)),
                    source_start_ms=0,
                    song_name=str(meta.get("song_name", "")),
                    artist_name=str(meta.get("artist_name", "")),
                )
            )
        if segments:
            last = segments[-1]
            segments[-1] = SegmentSpec(
                position=last.position,
                mix_item_id=last.mix_item_id,
                start_ms=last.start_ms,
                end_ms=int(round((total_len_n / self.RENDER_SR) * 1000)),
                source_start_ms=last.source_start_ms,
                song_name=last.song_name,
                artist_name=last.artist_name,
            )
        mix_audio = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
        return mix_audio, segments

    def _pick_outgoing_transition_start(
        self,
        *,
        default_transition_start_n: int,
        mix_len_n: int,
        overlap_n: int,
        outgoing_feat: _TrackFeatures,
        placed: _PlacedTrack,
    ) -> int:
        out = outgoing_feat.switch_out
        if out is None or not out.downbeats_s:
            return default_transition_start_n

        candidates: list[int] = []
        for d in out.downbeats_s:
            if d < placed.source_start_s:
                continue
            rel = (d - placed.source_start_s) / max(placed.stretch_rate, 1e-9)
            abs_s = placed.mix_start_s + rel
            idx = int(round(abs_s * self.RENDER_SR))
            if 0 <= idx <= max(0, mix_len_n - overlap_n):
                candidates.append(idx)
        if not candidates:
            return default_transition_start_n
        return min(candidates, key=lambda c: (abs(c - default_transition_start_n), c))

    def _incoming_source_start(self, feat: _TrackFeatures) -> float:
        sec = feat.switch_in
        if sec is None:
            return 0.0
        if sec.downbeats_s:
            return max(0.0, sec.downbeats_s[0])
        if sec.beats_s:
            return max(0.0, sec.beats_s[0])
        return max(0.0, sec.start_s)

    def _track_region(self, feat: _TrackFeatures | None) -> tuple[float, float] | None:
        """
        Return (start_s, end_s) region to render from the source track.
        Uses analyzer switch_in/out windows; returns None when not usable.
        """
        if feat is None or not feat.valid:
            return None
        if feat.switch_in is None or feat.switch_out is None:
            return None
        start_s = float(min(feat.switch_in.start_s, feat.switch_in.end_s))
        end_s = float(max(feat.switch_out.start_s, feat.switch_out.end_s))
        start_s = max(0.0, start_s - self.TRIM_PAD_PRE_S)
        end_s = max(start_s, end_s + self.TRIM_PAD_POST_S)
        if end_s - start_s < self.MIN_TRIMMED_WINDOW_S:
            return None
        return (start_s, end_s)

    def _equal_power_blend(self, out: np.ndarray, inn: np.ndarray) -> np.ndarray:
        n = min(len(out), len(inn))
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        theta = np.linspace(0.0, math.pi * 0.5, n, endpoint=True, dtype=np.float32)
        return (out[:n] * np.cos(theta)[:, None] + inn[:n] * np.sin(theta)[:, None]).astype(np.float32)

    def _apply_bass_swap(self, incoming_overlap: np.ndarray) -> np.ndarray:
        n = len(incoming_overlap)
        if n < 8:
            return incoming_overlap
        filtered = self._highpass(incoming_overlap, cutoff_hz=self.BASS_SWAP_HPF_HZ)
        out = incoming_overlap.copy()
        half = n // 2
        out[:half] = filtered[:half]
        if half < n:
            ramp = np.linspace(1.0, 0.0, n - half, endpoint=True, dtype=np.float32)[:, None]
            out[half:] = filtered[half:] * ramp + incoming_overlap[half:] * (1.0 - ramp)
        return out

    # ---------- Audio utils ----------

    def _highpass(self, audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
        wn = float(np.clip(cutoff_hz / (self.RENDER_SR * 0.5), 1e-4, 0.99))
        b, a = cast(tuple[np.ndarray, np.ndarray], butter(2, wn, btype="highpass", output="ba"))
        l_raw = lfilter(b, a, audio[:, 0])
        r_raw = lfilter(b, a, audio[:, 1])
        l = self._ensure_float_array(l_raw[0] if isinstance(l_raw, tuple) else l_raw)
        r = self._ensure_float_array(r_raw[0] if isinstance(r_raw, tuple) else r_raw)
        return np.column_stack([l, r]).astype(np.float32)

    def _load_audio_stereo(self, abs_path: str) -> np.ndarray:
        read_result = sf.read(abs_path, dtype="float32", always_2d=True)
        if not isinstance(read_result, tuple) or len(read_result) < 2:
            return np.zeros((0, 2), dtype=np.float32)
        y = cast(np.ndarray, read_result[0])
        sr = int(cast(int | float, read_result[1]))
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        elif y.shape[1] > 2:
            y = y[:, :2]

        if sr == self.RENDER_SR:
            return y.astype(np.float32)

        return self._resample_stereo(y, orig_sr=sr, target_sr=self.RENDER_SR)

    def _load_audio_stereo_region(self, abs_path: str, *, start_s: float, end_s: float | None) -> np.ndarray:
        """
        Load a time window [start_s, end_s) from a file as stereo float32 at RENDER_SR.

        Uses SoundFile seeking to avoid decoding the full track when possible.
        """
        start_s = float(max(0.0, start_s))
        try:
            with sf.SoundFile(abs_path) as f:
                sr = int(f.samplerate)
                total_frames = int(f.frames)
                start_frame = int(round(start_s * sr))
                start_frame = int(np.clip(start_frame, 0, max(0, total_frames)))

                if end_s is None:
                    end_frame = total_frames
                else:
                    end_s = float(max(start_s, end_s))
                    end_frame = int(round(end_s * sr))
                    end_frame = int(np.clip(end_frame, start_frame, total_frames))

                n_frames = max(0, end_frame - start_frame)
                if n_frames <= 0:
                    return np.zeros((0, 2), dtype=np.float32)

                try:
                    f.seek(start_frame)
                except Exception:
                    # Some formats have limited seeking support; fall back to full read.
                    return self._load_audio_stereo(abs_path)

                y = f.read(frames=n_frames, dtype="float32", always_2d=True)
        except Exception:
            return self._load_audio_stereo(abs_path)

        if y.ndim != 2 or y.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Ensure stereo (2ch max).
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        elif y.shape[1] > 2:
            y = y[:, :2]

        if sr == self.RENDER_SR:
            return y.astype(np.float32)

        return self._resample_stereo(y, orig_sr=sr, target_sr=self.RENDER_SR)

    def _resample_stereo(self, y: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 2 or y.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        elif y.shape[1] > 2:
            y = y[:, :2]
        if orig_sr == target_sr:
            return y.astype(np.float32)

        # Resample both channels in one call to reduce overhead and keep channels aligned.
        y_cf = y.T  # (2, n)
        y_rs = self._resample_audio(y_cf, orig_sr=orig_sr, target_sr=target_sr)
        y_rs = self._ensure_float_array(y_rs)
        if y_rs.ndim != 2 or y_rs.shape[0] < 2:
            return np.zeros((0, 2), dtype=np.float32)
        return y_rs[:2, :].T.astype(np.float32)

    def _resample_audio(self, y: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return self._ensure_float_array(y)

        res_types: list[str] = []
        preferred = str(self.resample_res_type or "").strip()
        if preferred:
            res_types.append(preferred)
        # Always include a sane fallback order.
        if "soxr_hq" not in res_types:
            res_types.append("soxr_hq")
        if "kaiser_fast" not in res_types:
            res_types.append("kaiser_fast")

        last_exc: Exception | None = None
        for res_type in res_types:
            try:
                out = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type)
                return self._ensure_float_array(out)
            except ModuleNotFoundError as exc:
                last_exc = exc
                msg = str(exc)
                if res_type.startswith("soxr_") and "soxr" in msg:
                    continue
                if res_type.startswith("kaiser") and "resampy" in msg:
                    continue
                raise

        # Last resort: polyphase resampling (deterministic).
        try:
            from scipy.signal import resample_poly  # type: ignore

            g = math.gcd(int(orig_sr), int(target_sr))
            up = int(target_sr) // g
            down = int(orig_sr) // g
            y_arr = np.asarray(y, dtype=np.float32)
            if y_arr.ndim == 1:
                return self._ensure_float_array(resample_poly(y_arr, up, down))
            rows = [self._ensure_float_array(resample_poly(row, up, down)) for row in y_arr]
            return np.stack(rows, axis=0).astype(np.float32)
        except Exception:
            if last_exc is not None:
                raise last_exc
            raise

    def _time_stretch_stereo(self, audio: np.ndarray, rate: float) -> np.ndarray:
        if audio.size == 0 or abs(rate - 1.0) <= 1e-5:
            return audio
        # Use one multi-channel call to reduce overhead and keep channels sample-aligned.
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim != 2 or y.shape[1] != 2:
            y = self._ensure_float_array(y).reshape((-1, 2)) if y.size else np.zeros((0, 2), dtype=np.float32)
        try:
            yt_raw = librosa.effects.time_stretch(
                y.T,
                rate=rate,
                n_fft=int(self.time_stretch_n_fft),
                hop_length=int(self.time_stretch_hop_length),
            )
        except MemoryError:
            self.logger.warning(
                "Librosa time_stretch OOM for %d samples; falling back to chunked stretching (rate=%.6f).",
                int(len(y)),
                float(rate),
            )
            return self._time_stretch_stereo_chunked(y, rate)
        yt = self._ensure_float_array(yt_raw)
        if yt.ndim != 2:
            return np.zeros((0, 2), dtype=np.float32)
        if yt.shape[0] < 2:
            yt = np.vstack([yt, yt])
        out = yt[:2, :].T
        return out.astype(np.float32)

    def _time_stretch_stereo_chunked(self, audio: np.ndarray, rate: float) -> np.ndarray:
        y = np.asarray(audio, dtype=np.float32)
        if y.size == 0 or abs(rate - 1.0) <= 1e-5:
            return y
        if y.ndim != 2 or y.shape[1] != 2:
            y = self._ensure_float_array(y).reshape((-1, 2)) if y.size else np.zeros((0, 2), dtype=np.float32)

        n_fft = int(self.time_stretch_n_fft)
        hop_length = int(self.time_stretch_hop_length)
        freq_bins = 1 + (n_fft // 2)

        # Keep per-chunk STFT buffers bounded to avoid OOM.
        # Rough peak allocation for phase-vocoder buffers is O(channels * freq_bins * frames).
        target_bytes = 48 * 1024 * 1024
        bytes_per_frame = 2 * freq_bins * 8  # complex64 per bin
        target_frames = max(256, int(target_bytes // max(1, bytes_per_frame)))
        chunk_in_n = max(hop_length * 8, int(target_frames * hop_length))
        chunk_in_n = int((chunk_in_n // hop_length) * hop_length)

        pad_in_n = min(int(round(0.25 * self.RENDER_SR)), max(0, chunk_in_n // 4))
        fade_n = int(round(0.02 * self.RENDER_SR))

        out_chunks: list[np.ndarray] = []
        start_in = 0
        total_in_n = int(len(y))
        while start_in < total_in_n:
            center_end_in = min(total_in_n, start_in + chunk_in_n)
            read_start_in = max(0, start_in - pad_in_n)
            read_end_in = min(total_in_n, center_end_in + pad_in_n)

            y_read = y[read_start_in:read_end_in].T
            try:
                stretched_raw = librosa.effects.time_stretch(
                    y_read,
                    rate=rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                )
            except MemoryError:
                # Reduce chunk size and retry deterministically.
                if chunk_in_n <= hop_length * 8:
                    raise
                chunk_in_n = int(max(hop_length * 8, chunk_in_n // 2))
                chunk_in_n = int((chunk_in_n // hop_length) * hop_length)
                continue

            stretched = self._ensure_float_array(stretched_raw)
            if stretched.ndim != 2:
                return np.zeros((0, 2), dtype=np.float32)
            if stretched.shape[0] < 2:
                stretched = np.vstack([stretched, stretched])

            keep_start_out = int(round((start_in - read_start_in) / rate))
            keep_end_out = int(round((center_end_in - read_start_in) / rate))
            keep_start_out = int(np.clip(keep_start_out, 0, stretched.shape[1]))
            keep_end_out = int(np.clip(keep_end_out, keep_start_out, stretched.shape[1]))
            chunk_out = stretched[:2, keep_start_out:keep_end_out].T.astype(np.float32, copy=False)

            if out_chunks and fade_n > 0 and len(out_chunks[-1]) and len(chunk_out):
                f = int(min(fade_n, len(out_chunks[-1]), len(chunk_out)))
                if f > 0:
                    out_chunks[-1][-f:] = self._equal_power_blend(out_chunks[-1][-f:], chunk_out[:f])
                    chunk_out = chunk_out[f:]

            if chunk_out.size:
                out_chunks.append(chunk_out)

            start_in = center_end_in

        if not out_chunks:
            return np.zeros((0, 2), dtype=np.float32)
        out = np.vstack(out_chunks) if len(out_chunks) > 1 else out_chunks[0]

        expected_n = int(round(total_in_n / rate))
        if expected_n <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(out) < expected_n:
            out = np.vstack([out, np.zeros((expected_n - len(out), 2), dtype=np.float32)])
        elif len(out) > expected_n:
            out = out[:expected_n]
        return out.astype(np.float32, copy=False)

    def _encode_output_bytes(self, audio: np.ndarray) -> bytes:
        mime = str(self.output_mime or self.DEFAULT_OUTPUT_MIME).lower().strip()
        if mime in ("audio/wav", "audio/wave", "audio/x-wav"):
            return self._float_stereo_to_wav_bytes(audio)
        if mime in ("audio/mpeg", "audio/mp3"):
            return self._float_stereo_to_mp3_bytes(audio)
        # Deterministic default.
        return self._float_stereo_to_wav_bytes(audio)

    def _float_stereo_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2:
            audio = np.zeros((1, 2), dtype=np.float32)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.999:
            audio = audio * (0.999 / peak)
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        raw = pcm.tobytes()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.SAMPWIDTH)
            wf.setframerate(self.RENDER_SR)
            wf.writeframes(raw)
        return buf.getvalue()

    def _float_stereo_to_mp3_bytes(self, audio: np.ndarray) -> bytes:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2:
            audio = np.zeros((1, 2), dtype=np.float32)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]

        # Keep levels bounded before lossy encoding.
        audio = np.clip(audio, -1.0, 1.0)
        buf = io.BytesIO()
        sf.write(buf, audio, self.RENDER_SR, format="MP3", subtype="MPEG_LAYER_III")
        return buf.getvalue()

    def _generate_silence_wav(self, duration_ms: int) -> bytes:
        frames = int(self.RENDER_SR * duration_ms / 1000)
        silence_frame = struct.pack("<hh", 0, 0)
        raw = silence_frame * frames
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.SAMPWIDTH)
            wf.setframerate(self.RENDER_SR)
            wf.writeframes(raw)
        return buf.getvalue()

    def _record_timing(self, debug: dict[str, Any], label: str, start: float) -> None:
        elapsed = float(time.perf_counter() - start)
        timing = debug.setdefault("timing_s", {})
        timing[label] = elapsed
        if self.enable_timing_logs:
            self.logger.info("Render timing %s: %.3fs", label, elapsed)

    def _plan_debug_payload(self, plans: list[_TransitionPlan]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for p in plans:
            payload.append(
                {
                    "from_id": str(p.from_id),
                    "to_id": str(p.to_id),
                    "incoming_bpm": float(p.option.incoming_bpm),
                    "target_bpm": float(p.option.target_bpm),
                    "delta_bpm_pct": float(p.option.delta_bpm_pct),
                    "stretch_rate": float(p.option.stretch_rate),
                    "secondary_cost": float(p.option.secondary_cost),
                    "total_cost": float(p.option.total_cost),
                    "overlap_bars": int(p.overlap_bars),
                    "overlap_s": float(p.overlap_s),
                }
            )
        return payload

    def _emit_render_debug(self, debug: dict[str, Any]) -> None:
        if self.enable_debug_logs:
            self.logger.info("Render debug payload: %s", debug.get("decisions", {}))

    # ---------- Parse helpers ----------

    def _to_optional_float(self, v: Any) -> float | None:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(f):
            return None
        return f

    def _to_float(self, v: Any, default: float = 0.0) -> float:
        f = self._to_optional_float(v)
        if f is None:
            return float(default)
        return f

    def _to_int(self, v: Any, default: int = 0) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _to_str(self, v: Any) -> str | None:
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None

    def _float_tuple(self, arr: Any) -> tuple[float, ...]:
        if not isinstance(arr, list):
            return ()
        out: list[float] = []
        for v in arr:
            f = self._to_optional_float(v)
            if f is not None:
                out.append(f)
        return tuple(out)

    def _ensure_float_array(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value.astype(np.float32)
        if isinstance(value, tuple):
            for item in value:
                if isinstance(item, np.ndarray):
                    return item.astype(np.float32)
        return np.asarray(value, dtype=np.float32)

    def _as_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}


class VariableBpmMixRenderer(DeterministicMixRenderer):
    TEMPO_RAMP_CHUNKS = 16
    HARD_TEMPO_MISMATCH_PENALTY = 400.0

    def __init__(
        self,
        *,
        enable_timing_logs: bool = False,
        enable_debug_logs: bool = False,
        output_mime: str | None = None,
        resample_res_type: str | None = None,
        time_stretch_n_fft: int = 1024,
        time_stretch_hop_length: int = 256,
        tempo_ramp_chunks: int = TEMPO_RAMP_CHUNKS,
    ) -> None:
        super().__init__(
            enable_timing_logs=enable_timing_logs,
            enable_debug_logs=enable_debug_logs,
            output_mime=output_mime,
            resample_res_type=resample_res_type,
            time_stretch_n_fft=time_stretch_n_fft,
            time_stretch_hop_length=time_stretch_hop_length,
        )
        self.tempo_ramp_chunks = max(2, int(tempo_ramp_chunks))

    def _build_transition_plan(
        self, ordered_tracks: list[TrackInput], features: dict[uuid.UUID, _TrackFeatures]
    ) -> list[_TransitionPlan]:
        plans: list[_TransitionPlan] = []
        for i in range(len(ordered_tracks) - 1):
            a = ordered_tracks[i]
            b = ordered_tracks[i + 1]
            option = self._best_transition_option(features[a.mix_item_id], features[b.mix_item_id])

            out_sec = features[a.mix_item_id].switch_out
            beats_per_bar = out_sec.beats_per_bar if out_sec is not None else 4
            bpm = option.target_bpm if option.target_bpm > 0 else 120.0
            bar_s = (60.0 / bpm) * beats_per_bar

            overlap_bars = 8
            if option.delta_bpm_pct > 4.0 or option.key_clash or option.signature_penalty > 0.55:
                overlap_bars = 4
            elif option.delta_bpm_pct < 1.5 and option.signature_penalty < 0.25 and not option.key_clash:
                overlap_bars = 12

            overlap_s = max(self.MIN_OVERLAP_S, min(self.MAX_OVERLAP_S, bar_s * overlap_bars))
            plans.append(
                _TransitionPlan(
                    from_id=a.mix_item_id,
                    to_id=b.mix_item_id,
                    option=option,
                    overlap_bars=overlap_bars,
                    overlap_s=overlap_s,
                )
            )
        return plans

    def _best_transition_option(
        self, a: _TrackFeatures, b: _TrackFeatures, *, target_bpm: float | None = None
    ) -> _TransitionOption:
        _ = target_bpm
        a_out = a.switch_out
        b_in = b.switch_in
        if a_out is None or b_in is None or a_out.bpm_local is None:
            fallback_bpm_a = a_out.bpm_local if a_out and a_out.bpm_local else 120.0
            fallback_bpm_b = b_in.bpm_local if b_in and b_in.bpm_local else fallback_bpm_a
            delta_raw = abs(fallback_bpm_b - fallback_bpm_a) / max(fallback_bpm_a, 1e-9) * 100.0
            start_raw, start_clamped, start_err, _, _, end_err = self._rate_metrics(
                bpm_a=fallback_bpm_a, bpm_b=fallback_bpm_b
            )
            return _TransitionOption(
                incoming_bpm=fallback_bpm_b,
                target_bpm=fallback_bpm_a,
                delta_bpm_pct=delta_raw,
                stretch_rate=start_clamped,
                key_penalty=0.5,
                signature_penalty=0.5,
                instability_penalty=0.0,
                secondary_cost=2.0,
                total_cost=9999.0 + start_err + end_err + abs(start_raw - start_clamped),
                key_clash=False,
            )

        bpm_a = float(a_out.bpm_local)
        incoming = list(b_in.bpm_candidates)
        if b_in.bpm_local is not None:
            incoming.append(float(b_in.bpm_local))
        incoming = sorted({round(v, 6) for v in incoming if v > 0})
        if not incoming:
            incoming = [bpm_a]

        options: list[_TransitionOption] = []
        for bpm_b in incoming:
            delta_raw_pct = abs(bpm_b - bpm_a) / max(bpm_a, 1e-9) * 100.0
            start_raw, start_rate, start_err, end_raw, _, end_err = self._rate_metrics(bpm_a=bpm_a, bpm_b=bpm_b)
            infeasible = (
                start_raw < self.MIN_STRETCH_RATE
                or start_raw > self.MAX_STRETCH_RATE
                or end_raw < self.MIN_STRETCH_RATE
                or end_raw > self.MAX_STRETCH_RATE
            )

            tempo_cost = delta_raw_pct + (start_err + end_err) * 40.0
            if infeasible:
                tempo_cost += self.HARD_TEMPO_MISMATCH_PENALTY

            key_pen = self._key_penalty(a_out.key, b_in.key)
            key_weight = min(max(a_out.key.confidence, 0.0), max(b_in.key.confidence, 0.0))
            key_pen_w = key_pen * key_weight
            sig_pen = self._signature_penalty(a_out, b_in)
            timbre_pen = self._timbre_penalty(a_out, b_in)
            sec = 0.8 * key_pen_w + 1.0 * sig_pen + 1.0 * timbre_pen
            total = tempo_cost * 100.0 + sec * 10.0

            options.append(
                _TransitionOption(
                    incoming_bpm=bpm_b,
                    target_bpm=bpm_a,
                    delta_bpm_pct=delta_raw_pct,
                    stretch_rate=start_rate,
                    key_penalty=key_pen_w,
                    signature_penalty=sig_pen,
                    instability_penalty=0.0,
                    secondary_cost=sec,
                    total_cost=total,
                    key_clash=(key_pen_w >= 0.55),
                )
            )

        options.sort(
            key=lambda o: (
                o.total_cost,
                o.delta_bpm_pct,
                o.secondary_cost,
                o.incoming_bpm,
                str(b.track.mix_item_id),
            )
        )
        return options[0]

    def _render_with_plan(
        self,
        ordered_tracks: list[TrackInput],
        features: dict[uuid.UUID, _TrackFeatures],
        plans: list[_TransitionPlan],
    ) -> tuple[np.ndarray, list[SegmentSpec]]:
        first = ordered_tracks[0]
        first_source_start_s = 0.0
        first_audio = self._load_audio_stereo(first.abs_path)
        chunks: list[np.ndarray] = [self._sanitize_stereo(first_audio)]
        total_len_n = int(len(chunks[0]))

        first_meta = first.metadata_json or {}
        segments: list[SegmentSpec] = [
            SegmentSpec(
                position=0,
                mix_item_id=first.mix_item_id,
                start_ms=0,
                end_ms=int(round((total_len_n / self.RENDER_SR) * 1000)),
                source_start_ms=int(round(first_source_start_s * 1000)),
                song_name=str(first_meta.get("song_name", "")),
                artist_name=str(first_meta.get("artist_name", "")),
            )
        ]
        placed = _PlacedTrack(track=first, source_start_s=first_source_start_s, stretch_rate=1.0, mix_start_s=0.0)

        for i, plan in enumerate(plans):
            incoming_track = ordered_tracks[i + 1]
            incoming_feat = features[incoming_track.mix_item_id]
            outgoing_feat = features[placed.track.mix_item_id]

            region = self._track_region(incoming_feat)
            region_start_s = region[0] if region is not None else 0.0
            region_end_s = region[1] if region is not None else None

            source_start_s = max(region_start_s, self._incoming_source_start(incoming_feat))
            incoming = (
                self._load_audio_stereo_region(incoming_track.abs_path, start_s=source_start_s, end_s=region_end_s)
                if region_end_s is not None
                else self._load_audio_stereo(incoming_track.abs_path)
            )
            incoming = self._sanitize_stereo(incoming)
            if incoming.size == 0:
                source_start_s = 0.0
                incoming = self._sanitize_stereo(self._load_audio_stereo(incoming_track.abs_path))

            out_chunk = self._sanitize_stereo(chunks[-1])
            chunks[-1] = out_chunk
            prefix_len_n = total_len_n - int(len(out_chunk))
            out_start_n = prefix_len_n

            overlap_n = int(round(plan.overlap_s * self.RENDER_SR))
            overlap_n = int(np.clip(overlap_n, 1, max(1, len(out_chunk) - 1)))

            out_fallback = outgoing_feat.switch_out.bpm_local if outgoing_feat.switch_out else None
            in_fallback = incoming_feat.switch_in.bpm_local if incoming_feat.switch_in else None
            bpm_a = (
                float(plan.option.target_bpm)
                if plan.option.target_bpm > 0
                else float(out_fallback)
                if out_fallback is not None and out_fallback > 0
                else 120.0
            )
            bpm_b = (
                float(plan.option.incoming_bpm)
                if plan.option.incoming_bpm > 0
                else float(in_fallback)
                if in_fallback is not None and in_fallback > 0
                else bpm_a
            )

            transition_start_n = out_start_n
            start_in_chunk = 0
            out_needed_n = 1
            for _ in range(6):
                overlap_n = int(np.clip(overlap_n, 1, max(1, len(out_chunk) - 1)))
                out_needed_n = self._estimate_source_samples(
                    bpm_native=bpm_a,
                    bpm_start=bpm_a,
                    bpm_end=bpm_b,
                    output_n=overlap_n,
                )
                max_start_n = total_len_n - max(1, out_needed_n)
                if max_start_n < out_start_n:
                    if overlap_n <= 1:
                        break
                    ratio = max(0.5, (len(out_chunk) - 1) / max(1, out_needed_n))
                    overlap_n = max(1, int(math.floor(overlap_n * ratio)))
                    continue

                default_start_n = max(out_start_n, total_len_n - overlap_n)
                picked = self._pick_outgoing_transition_start(
                    default_transition_start_n=default_start_n,
                    mix_len_n=total_len_n,
                    overlap_n=overlap_n,
                    outgoing_feat=outgoing_feat,
                    placed=placed,
                )
                transition_start_n = int(np.clip(picked, out_start_n, max_start_n))
                start_in_chunk = transition_start_n - out_start_n
                available = len(out_chunk) - start_in_chunk
                if out_needed_n <= max(1, available):
                    break
                if overlap_n <= 1:
                    break
                ratio = max(0.5, max(1, available) / max(1, out_needed_n))
                overlap_n = max(1, int(math.floor(overlap_n * ratio)))

            start_in_chunk = int(np.clip(start_in_chunk, 0, max(0, len(out_chunk) - 1)))
            out_source = out_chunk[start_in_chunk:]

            out_ov_warped, _ = self._warp_overlap_piecewise(
                source=out_source,
                bpm_native=bpm_a,
                bpm_start=bpm_a,
                bpm_end=bpm_b,
                output_n=overlap_n,
            )
            in_ov_warped, consumed_in_n = self._warp_overlap_piecewise(
                source=incoming,
                bpm_native=bpm_b,
                bpm_start=bpm_a,
                bpm_end=bpm_b,
                output_n=overlap_n,
            )
            consumed_in_n = int(np.clip(consumed_in_n, 0, len(incoming)))

            blend = self._equal_power_blend(out_ov_warped, self._apply_bass_swap(in_ov_warped))
            new_out_chunk = np.vstack([out_chunk[:start_in_chunk], blend]).astype(np.float32)
            chunks[-1] = new_out_chunk

            tail = incoming[consumed_in_n:]
            if tail.size:
                chunks.append(tail.astype(np.float32))

            total_len_n = prefix_len_n + int(len(new_out_chunk)) + int(len(tail))

            prev = segments[-1]
            segments[-1] = SegmentSpec(
                position=prev.position,
                mix_item_id=prev.mix_item_id,
                start_ms=prev.start_ms,
                end_ms=int(round(((transition_start_n + overlap_n) / self.RENDER_SR) * 1000)),
                source_start_ms=prev.source_start_ms,
                song_name=prev.song_name,
                artist_name=prev.artist_name,
            )

            meta = incoming_track.metadata_json or {}
            segments.append(
                SegmentSpec(
                    position=i + 1,
                    mix_item_id=incoming_track.mix_item_id,
                    start_ms=int(round((transition_start_n / self.RENDER_SR) * 1000)),
                    end_ms=int(round((total_len_n / self.RENDER_SR) * 1000)),
                    source_start_ms=int(round(source_start_s * 1000)),
                    song_name=str(meta.get("song_name", "")),
                    artist_name=str(meta.get("artist_name", "")),
                )
            )

            source_shift_s = (consumed_in_n - overlap_n) / self.RENDER_SR
            placed = _PlacedTrack(
                track=incoming_track,
                source_start_s=source_start_s + source_shift_s,
                stretch_rate=1.0,
                mix_start_s=transition_start_n / self.RENDER_SR,
            )

        if segments:
            last = segments[-1]
            segments[-1] = SegmentSpec(
                position=last.position,
                mix_item_id=last.mix_item_id,
                start_ms=last.start_ms,
                end_ms=int(round((total_len_n / self.RENDER_SR) * 1000)),
                source_start_ms=last.source_start_ms,
                song_name=last.song_name,
                artist_name=last.artist_name,
            )
        mix_audio = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
        return mix_audio.astype(np.float32), segments

    def _plan_debug_payload(self, plans: list[_TransitionPlan]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for p in plans:
            bpm_a = float(p.option.target_bpm) if p.option.target_bpm > 0 else 120.0
            bpm_b = float(p.option.incoming_bpm) if p.option.incoming_bpm > 0 else bpm_a
            start_raw, start_rate, start_err, end_raw, end_rate, end_err = self._rate_metrics(
                bpm_a=bpm_a, bpm_b=bpm_b
            )
            payload.append(
                {
                    "from_id": str(p.from_id),
                    "to_id": str(p.to_id),
                    "bpm_a": bpm_a,
                    "bpm_b": bpm_b,
                    "delta_raw_bpm_pct": float(p.option.delta_bpm_pct),
                    "start_rate_raw": float(start_raw),
                    "start_rate_clamped": float(start_rate),
                    "start_match_error_pct": float(start_err),
                    "end_rate_raw": float(end_raw),
                    "end_rate_clamped": float(end_rate),
                    "end_match_error_pct": float(end_err),
                    "secondary_cost": float(p.option.secondary_cost),
                    "total_cost": float(p.option.total_cost),
                    "overlap_bars": int(p.overlap_bars),
                    "overlap_s": float(p.overlap_s),
                    "tempo_ramp_chunks": int(self.tempo_ramp_chunks),
                }
            )
        return payload

    def _rate_metrics(self, *, bpm_a: float, bpm_b: float) -> tuple[float, float, float, float, float, float]:
        safe_a = max(float(bpm_a), 1e-9)
        safe_b = max(float(bpm_b), 1e-9)
        start_raw = safe_a / safe_b
        end_raw = safe_b / safe_a
        start_clamped = float(np.clip(start_raw, self.MIN_STRETCH_RATE, self.MAX_STRETCH_RATE))
        end_clamped = float(np.clip(end_raw, self.MIN_STRETCH_RATE, self.MAX_STRETCH_RATE))
        start_eff_bpm = safe_b * start_clamped
        end_eff_bpm = safe_a * end_clamped
        start_err_pct = abs(start_eff_bpm - safe_a) / safe_a * 100.0
        end_err_pct = abs(end_eff_bpm - safe_b) / safe_b * 100.0
        return start_raw, start_clamped, start_err_pct, end_raw, end_clamped, end_err_pct

    def _warp_overlap_piecewise(
        self,
        *,
        source: np.ndarray,
        bpm_native: float,
        bpm_start: float,
        bpm_end: float,
        output_n: int,
    ) -> tuple[np.ndarray, int]:
        output_n = int(max(0, output_n))
        if output_n <= 0:
            return np.zeros((0, 2), dtype=np.float32), 0

        src = self._sanitize_stereo(source)
        chunk_lengths = self._split_lengths(output_n, self.tempo_ramp_chunks)
        if not chunk_lengths:
            return np.zeros((0, 2), dtype=np.float32), 0

        rendered: list[np.ndarray] = []
        consumed = 0
        n_chunks = len(chunk_lengths)
        for idx, out_len in enumerate(chunk_lengths):
            frac = idx / (n_chunks - 1) if n_chunks > 1 else 1.0
            bpm_target = bpm_start + (bpm_end - bpm_start) * frac
            raw_rate = bpm_target / max(float(bpm_native), 1e-9)
            rate = float(np.clip(raw_rate, self.MIN_STRETCH_RATE, self.MAX_STRETCH_RATE))
            src_len = max(1, int(round(out_len * rate)))

            raw_chunk = src[consumed : consumed + src_len]
            if len(raw_chunk) < src_len:
                pad_n = src_len - len(raw_chunk)
                raw_chunk = np.vstack([raw_chunk, np.zeros((pad_n, 2), dtype=np.float32)])

            rendered_chunk = self._stretch_to_length(raw_chunk, rate=rate, target_n=out_len)
            rendered.append(rendered_chunk)
            consumed += src_len

        warped = np.vstack(rendered) if len(rendered) > 1 else rendered[0]
        warped = self._fit_length(warped, output_n)
        consumed_clamped = int(np.clip(consumed, 0, len(src)))
        return warped, consumed_clamped

    def _estimate_source_samples(self, *, bpm_native: float, bpm_start: float, bpm_end: float, output_n: int) -> int:
        output_n = int(max(0, output_n))
        if output_n <= 0:
            return 0
        chunk_lengths = self._split_lengths(output_n, self.tempo_ramp_chunks)
        n_chunks = len(chunk_lengths)
        total = 0
        for idx, out_len in enumerate(chunk_lengths):
            frac = idx / (n_chunks - 1) if n_chunks > 1 else 1.0
            bpm_target = bpm_start + (bpm_end - bpm_start) * frac
            raw_rate = bpm_target / max(float(bpm_native), 1e-9)
            rate = float(np.clip(raw_rate, self.MIN_STRETCH_RATE, self.MAX_STRETCH_RATE))
            total += max(1, int(round(out_len * rate)))
        return int(max(1, total))

    def _split_lengths(self, total: int, parts: int) -> list[int]:
        total = int(max(0, total))
        parts = int(max(1, parts))
        if total <= 0:
            return []
        base = total // parts
        rem = total % parts
        lengths = [base + (1 if i < rem else 0) for i in range(parts)]
        return [x for x in lengths if x > 0]

    def _stretch_to_length(self, audio: np.ndarray, *, rate: float, target_n: int) -> np.ndarray:
        src = self._sanitize_stereo(audio)
        if target_n <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(src) == 0:
            return np.zeros((target_n, 2), dtype=np.float32)
        if abs(rate - 1.0) <= 1e-5 and len(src) == target_n:
            return src
        min_for_phase = max(32, int(self.time_stretch_n_fft // 2))
        if len(src) < min_for_phase:
            return self._resample_to_length(src, target_n)
        try:
            stretched = self._time_stretch_stereo(src, rate)
        except Exception:
            return self._resample_to_length(src, target_n)
        if stretched.size == 0:
            return self._resample_to_length(src, target_n)
        return self._fit_length(stretched, target_n)

    def _fit_length(self, audio: np.ndarray, target_n: int) -> np.ndarray:
        y = self._sanitize_stereo(audio)
        target_n = int(max(0, target_n))
        if target_n <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(y) == target_n:
            return y
        if len(y) > target_n:
            return y[:target_n]
        pad = np.zeros((target_n - len(y), 2), dtype=np.float32)
        return np.vstack([y, pad]).astype(np.float32)

    def _resample_to_length(self, audio: np.ndarray, target_n: int) -> np.ndarray:
        y = self._sanitize_stereo(audio)
        target_n = int(max(0, target_n))
        if target_n <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(y) <= 1:
            return np.zeros((target_n, 2), dtype=np.float32)
        if len(y) == target_n:
            return y
        src_x = np.linspace(0.0, float(len(y) - 1), num=len(y), dtype=np.float64)
        dst_x = np.linspace(0.0, float(len(y) - 1), num=target_n, dtype=np.float64)
        left = np.interp(dst_x, src_x, y[:, 0].astype(np.float64))
        right = np.interp(dst_x, src_x, y[:, 1].astype(np.float64))
        return np.column_stack([left, right]).astype(np.float32)

    def _sanitize_stereo(self, audio: np.ndarray) -> np.ndarray:
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim != 2:
            return np.zeros((0, 2), dtype=np.float32)
        if y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        elif y.shape[1] > 2:
            y = y[:, :2]
        return y.astype(np.float32)


class PlaceholderMixRenderer:
    """
    Placeholder renderer:
    - makes a silent WAV
    - assigns each track a fixed duration (e.g., 3000ms)
    - produces sequential segments
    """

    PER_TRACK_MS = 3000
    SAMPLE_RATE = 44100
    CHANNELS = 2
    SAMPWIDTH = 2

    async def render(self, tracks: list[TrackInput], fixed_tracks: list[TrackInput]) -> RenderResult:
        n = len(tracks)
        length_ms = max(1, n) * self.PER_TRACK_MS
        segments: list[SegmentSpec] = []
        t = 0
        for i, tr in enumerate(tracks):
            meta = tr.metadata_json or {}
            segments.append(
                SegmentSpec(
                    position=i,
                    mix_item_id=tr.mix_item_id,
                    start_ms=t,
                    end_ms=t + self.PER_TRACK_MS,
                    source_start_ms=0,
                    song_name=str(meta.get("song_name", "")),
                    artist_name=str(meta.get("artist_name", "")),
                )
            )
            t += self.PER_TRACK_MS

        wav_bytes = self._generate_silence_wav(length_ms)
        return RenderResult(audio_bytes=wav_bytes, mime="audio/wav", length_ms=length_ms, segments=segments)

    def _generate_silence_wav(self, duration_ms: int) -> bytes:
        frames = int(self.SAMPLE_RATE * duration_ms / 1000)
        silence_frame = struct.pack("<hh", 0, 0)
        raw = silence_frame * frames
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.SAMPWIDTH)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(raw)
        return buf.getvalue()
