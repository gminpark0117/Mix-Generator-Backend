Auto-Mix analyzer output is designed to drive beatmatching and transition decisions. The schema is local-first: most descriptors are computed on switch-in/out sections rather than the full track.

Field summary
- schema_version: Schema version integer. Current value is 2.
- track.duration_s: Total duration of the track in seconds.
- track.sr: Analysis sample rate used for feature extraction.
- track.hop_length: Hop length used for time-based features.
- track.overview: Minimal global overview block.
- track.overview.global_bpm_guess: Full-track BPM estimate from the global beat grid.
- sections.switch_in.start_s: Start time of the switch-in window in seconds.
- sections.switch_in.end_s: End time of the switch-in window in seconds.
- sections.switch_in.quality.score: Window quality score used for selection.
- sections.switch_in.quality.components.onset_clarity: Percussive onset clarity in window.
- sections.switch_in.quality.components.tempo_stability: Stability score from IBI CV.
- sections.switch_in.quality.components.energy_ramp: Positive RMS slope penalty.
- sections.switch_in.quality.components.spectral_flux: Spectral flux penalty.
- sections.switch_in.quality.components.chroma_flux: Chroma flux penalty.
- sections.switch_in.rhythm.beats_s: Beat times in seconds for this section.
- sections.switch_in.rhythm.bpm_local: BPM estimate from section beats.
- sections.switch_in.rhythm.bpm_candidates: Half/double-time candidates.
- sections.switch_in.rhythm.meter.beats_per_bar: Estimated beats per bar.
- sections.switch_in.rhythm.meter.note_value: Meter note value (fixed at 4).
- sections.switch_in.rhythm.meter.confidence: Meter confidence score.
- sections.switch_in.rhythm.meter.assumed: Whether the meter is a fallback assumption.
- sections.switch_in.rhythm.downbeats_s: Downbeat times in seconds.
- sections.switch_in.rhythm.beat_positions: Beat position within bar (1..beats_per_bar).
- sections.switch_in.rhythm.beat_interval_s.mean_s: Mean inter-beat interval in seconds.
- sections.switch_in.rhythm.beat_interval_s.std_s: Standard deviation of intervals.
- sections.switch_in.rhythm.beat_interval_s.cv: Coefficient of variation of intervals.
- sections.switch_in.rhythm.beat_interval_s.bpm_std: Standard deviation of BPM estimates.
- sections.switch_in.rhythm.signature.onset_source: Onset source used for signature.
- sections.switch_in.rhythm.signature.periodicity.tempogram_top_peaks: List of {bpm, strength} peaks.
- sections.switch_in.rhythm.signature.periodicity.tempogram_profile_ds: Downsampled tempogram profile.
- sections.switch_in.rhythm.signature.phase.phase_bins_per_bar: Bin count for bar-phase pattern.
- sections.switch_in.rhythm.signature.phase.pattern: Normalized bar-phase pattern.
- sections.switch_in.rhythm.signature.phase.fft_mag: FFT magnitudes of the pattern.
- sections.switch_in.rhythm.signature.phase.stability.bar_to_bar_var: Variance across bars.
- sections.switch_in.rhythm.signature.phase.stability.n_bars_used: Bars used for phase aggregation.
- sections.switch_in.rhythm.signature.phase.tempo_scale_used: Tempo scale hypothesis used.
- sections.switch_in.rhythm.signature.phase.bar_offset_used: Beat offset used for downbeat alignment.
- sections.switch_in.harmony.chroma_mean: Mean chroma vector for the section.
- sections.switch_in.harmony.chroma_std: Chroma standard deviation.
- sections.switch_in.harmony.key.tonic: Estimated tonic (C..B).
- sections.switch_in.harmony.key.mode: Estimated mode (major/minor).
- sections.switch_in.harmony.key.confidence: Key confidence score.
- sections.switch_in.harmony.source.feature: Chroma feature used.
- sections.switch_in.harmony.source.hop_length: Hop length used for chroma.
- sections.switch_in.timbre.mfcc_mean: Mean MFCC vector.
- sections.switch_in.timbre.mfcc_std: MFCC standard deviation vector.
- sections.switch_in.timbre.spectral_flatness_mean: Mean spectral flatness.
- sections.switch_in.timbre.spectral_flatness_std: Spectral flatness std.
- sections.switch_in.timbre.zero_crossing_rate_mean: Mean ZCR.
- sections.switch_in.timbre.zero_crossing_rate_std: ZCR std.
- sections.switch_in.spectral.centroid_mean: Mean spectral centroid.
- sections.switch_in.spectral.centroid_std: Spectral centroid std.
- sections.switch_in.spectral.bandwidth_mean: Mean spectral bandwidth.
- sections.switch_in.spectral.bandwidth_std: Spectral bandwidth std.
- sections.switch_in.spectral.rolloff_mean: Mean spectral rolloff.
- sections.switch_in.spectral.rolloff_std: Spectral rolloff std.
- sections.switch_in.energy: Reserved object for backward compatibility. Currently `{}`.
- sections.switch_out.start_s: Start time of the switch-out window in seconds.
- sections.switch_out.end_s: End time of the switch-out window in seconds.
- sections.switch_out.quality.score: Window quality score used for selection.
- sections.switch_out.quality.components.onset_clarity: Percussive onset clarity in window.
- sections.switch_out.quality.components.tempo_stability: Stability score from IBI CV.
- sections.switch_out.quality.components.energy_ramp: Positive RMS slope penalty.
- sections.switch_out.quality.components.spectral_flux: Spectral flux penalty.
- sections.switch_out.quality.components.chroma_flux: Chroma flux penalty.
- sections.switch_out.rhythm.beats_s: Beat times in seconds for this section.
- sections.switch_out.rhythm.bpm_local: BPM estimate from section beats.
- sections.switch_out.rhythm.bpm_candidates: Half/double-time candidates.
- sections.switch_out.rhythm.meter.beats_per_bar: Estimated beats per bar.
- sections.switch_out.rhythm.meter.note_value: Meter note value (fixed at 4).
- sections.switch_out.rhythm.meter.confidence: Meter confidence score.
- sections.switch_out.rhythm.meter.assumed: Whether the meter is a fallback assumption.
- sections.switch_out.rhythm.downbeats_s: Downbeat times in seconds.
- sections.switch_out.rhythm.beat_positions: Beat position within bar (1..beats_per_bar).
- sections.switch_out.rhythm.beat_interval_s.mean_s: Mean inter-beat interval in seconds.
- sections.switch_out.rhythm.beat_interval_s.std_s: Standard deviation of intervals.
- sections.switch_out.rhythm.beat_interval_s.cv: Coefficient of variation of intervals.
- sections.switch_out.rhythm.beat_interval_s.bpm_std: Standard deviation of BPM estimates.
- sections.switch_out.rhythm.signature.onset_source: Onset source used for signature.
- sections.switch_out.rhythm.signature.periodicity.tempogram_top_peaks: List of {bpm, strength} peaks.
- sections.switch_out.rhythm.signature.periodicity.tempogram_profile_ds: Downsampled tempogram profile.
- sections.switch_out.rhythm.signature.phase.phase_bins_per_bar: Bin count for bar-phase pattern.
- sections.switch_out.rhythm.signature.phase.pattern: Normalized bar-phase pattern.
- sections.switch_out.rhythm.signature.phase.fft_mag: FFT magnitudes of the pattern.
- sections.switch_out.rhythm.signature.phase.stability.bar_to_bar_var: Variance across bars.
- sections.switch_out.rhythm.signature.phase.stability.n_bars_used: Bars used for phase aggregation.
- sections.switch_out.rhythm.signature.phase.tempo_scale_used: Tempo scale hypothesis used.
- sections.switch_out.rhythm.signature.phase.bar_offset_used: Beat offset used for downbeat alignment.
- sections.switch_out.harmony.chroma_mean: Mean chroma vector for the section.
- sections.switch_out.harmony.chroma_std: Chroma standard deviation.
- sections.switch_out.harmony.key.tonic: Estimated tonic (C..B).
- sections.switch_out.harmony.key.mode: Estimated mode (major/minor).
- sections.switch_out.harmony.key.confidence: Key confidence score.
- sections.switch_out.harmony.source.feature: Chroma feature used.
- sections.switch_out.harmony.source.hop_length: Hop length used for chroma.
- sections.switch_out.timbre.mfcc_mean: Mean MFCC vector.
- sections.switch_out.timbre.mfcc_std: MFCC standard deviation vector.
- sections.switch_out.timbre.spectral_flatness_mean: Mean spectral flatness.
- sections.switch_out.timbre.spectral_flatness_std: Spectral flatness std.
- sections.switch_out.timbre.zero_crossing_rate_mean: Mean ZCR.
- sections.switch_out.timbre.zero_crossing_rate_std: ZCR std.
- sections.switch_out.spectral.centroid_mean: Mean spectral centroid.
- sections.switch_out.spectral.centroid_std: Spectral centroid std.
- sections.switch_out.spectral.bandwidth_mean: Mean spectral bandwidth.
- sections.switch_out.spectral.bandwidth_std: Spectral bandwidth std.
- sections.switch_out.spectral.rolloff_mean: Mean spectral rolloff.
- sections.switch_out.spectral.rolloff_std: Spectral rolloff std.
- sections.switch_out.energy: Reserved object for backward compatibility. Currently `{}`.
- debug.params.energy_curve_points: Retained parameter for backward compatibility (not emitted as section energy).
- debug.params.analysis_sr: Analysis sample rate used.
- debug.params.hop_length: Hop length used.
- debug.params.phase_bins_per_bar: Phase bins per bar configured.
- debug.params.tempogram_peaks: Number of tempogram peaks stored.
- debug.fallbacks: List of fallback decisions taken.
- debug.decisions.switch_in.window_start_s: Selected switch-in window start.
- debug.decisions.switch_in.window_end_s: Selected switch-in window end.
- debug.decisions.switch_in.quality: Quality payload for window selection.
- debug.decisions.switch_out.window_start_s: Selected switch-out window start.
- debug.decisions.switch_out.window_end_s: Selected switch-out window end.
- debug.decisions.switch_out.quality: Quality payload for window selection.
- debug.errors: List of errors encountered during analysis.

Example analysis_json
```json
{
  "schema_version": 2,
  "track": {
    "duration_s": 214.6,
    "sr": 22050,
    "hop_length": 512,
    "overview": {
      "global_bpm_guess": 128.4
    }
  },
  "sections": {
    "switch_in": {
      "start_s": 8.2,
      "end_s": 24.5,
      "quality": {
        "score": 1.42,
        "components": {
          "onset_clarity": 1.18,
          "tempo_stability": 0.91,
          "energy_ramp": 0.08,
          "spectral_flux": 0.34,
          "chroma_flux": 0.29
        }
      },
      "rhythm": {
        "beats_s": [8.23, 8.70, 9.16, 9.63, 10.10, 10.56, 11.03, 11.50],
        "bpm_local": 128.7,
        "bpm_candidates": [128.7, 64.35, 257.4],
        "meter": {"beats_per_bar": 4, "note_value": 4, "confidence": 0.62, "assumed": false},
        "downbeats_s": [8.23, 10.10],
        "beat_positions": [1, 2, 3, 4, 1, 2, 3, 4],
        "beat_interval_s": {"mean_s": 0.466, "std_s": 0.012, "cv": 0.026, "bpm_std": 1.8},
        "signature": {
          "onset_source": "hpss_percussive",
          "periodicity": {
            "tempogram_top_peaks": [
              {"bpm": 128.0, "strength": 0.91},
              {"bpm": 64.0, "strength": 0.42},
              {"bpm": 192.0, "strength": 0.31}
            ],
            "tempogram_profile_ds": [0.12, 0.34, 0.58, 0.71, 0.52, 0.29, 0.15, 0.08]
          },
          "phase": {
            "phase_bins_per_bar": 16,
            "pattern": [0.0, 0.18, 0.62, 0.22, 0.05, 0.14, 0.47, 0.19, 0.03, 0.16, 0.58, 0.21, 0.04, 0.12, 0.44, 0.17],
            "fft_mag": [1.92, 0.74, 0.51, 0.33, 0.21, 0.18, 0.11, 0.09],
            "stability": {"bar_to_bar_var": 0.014, "n_bars_used": 6},
            "tempo_scale_used": 1.0,
            "bar_offset_used": 0
          }
        }
      },
      "harmony": {
        "chroma_mean": [0.32, 0.08, 0.11, 0.07, 0.19, 0.14, 0.06, 0.22, 0.09, 0.18, 0.05, 0.09],
        "chroma_std": [0.08, 0.03, 0.04, 0.02, 0.06, 0.05, 0.02, 0.07, 0.03, 0.05, 0.02, 0.03],
        "key": {"tonic": "E", "mode": "minor", "confidence": 0.53},
        "source": {"feature": "chroma_cqt", "hop_length": 512}
      },
      "timbre": {
        "mfcc_mean": [-102.1, 24.5, 18.2, 9.7, 1.3, -4.9, -7.2, -8.5, -5.6, -2.4, 0.8, 2.1, 1.4],
        "mfcc_std": [6.2, 3.8, 3.1, 2.7, 2.4, 2.2, 2.0, 1.8, 1.7, 1.5, 1.4, 1.3, 1.2],
        "spectral_flatness_mean": 0.18,
        "spectral_flatness_std": 0.04,
        "zero_crossing_rate_mean": 0.043,
        "zero_crossing_rate_std": 0.009
      },
      "spectral": {
        "centroid_mean": 2190.4,
        "centroid_std": 410.2,
        "bandwidth_mean": 1320.7,
        "bandwidth_std": 190.6,
        "rolloff_mean": 4920.1,
        "rolloff_std": 620.8
      },
      "energy": {}
    },
    "switch_out": {
      "start_s": 176.4,
      "end_s": 193.2,
      "quality": {
        "score": 1.36,
        "components": {
          "onset_clarity": 1.09,
          "tempo_stability": 0.94,
          "energy_ramp": 0.04,
          "spectral_flux": 0.31,
          "chroma_flux": 0.33
        }
      },
      "rhythm": {
        "beats_s": [176.45, 176.92, 177.39, 177.86, 178.32, 178.79, 179.26, 179.72],
        "bpm_local": 127.9,
        "bpm_candidates": [127.9, 63.95, 255.8],
        "meter": {"beats_per_bar": 4, "note_value": 4, "confidence": 0.58, "assumed": false},
        "downbeats_s": [176.45, 178.32],
        "beat_positions": [1, 2, 3, 4, 1, 2, 3, 4],
        "beat_interval_s": {"mean_s": 0.469, "std_s": 0.013, "cv": 0.028, "bpm_std": 1.9},
        "signature": {
          "onset_source": "hpss_percussive",
          "periodicity": {
            "tempogram_top_peaks": [
              {"bpm": 128.0, "strength": 0.88},
              {"bpm": 64.0, "strength": 0.39},
              {"bpm": 192.0, "strength": 0.28}
            ],
            "tempogram_profile_ds": [0.11, 0.33, 0.55, 0.69, 0.50, 0.27, 0.14, 0.07]
          },
          "phase": {
            "phase_bins_per_bar": 16,
            "pattern": [0.0, 0.16, 0.58, 0.24, 0.06, 0.13, 0.45, 0.21, 0.04, 0.15, 0.54, 0.23, 0.05, 0.11, 0.42, 0.19],
            "fft_mag": [1.84, 0.71, 0.49, 0.31, 0.19, 0.16, 0.10, 0.08],
            "stability": {"bar_to_bar_var": 0.016, "n_bars_used": 5},
            "tempo_scale_used": 1.0,
            "bar_offset_used": 0
          }
        }
      },
      "harmony": {
        "chroma_mean": [0.31, 0.07, 0.10, 0.06, 0.20, 0.15, 0.05, 0.23, 0.08, 0.19, 0.05, 0.09],
        "chroma_std": [0.07, 0.03, 0.04, 0.02, 0.06, 0.05, 0.02, 0.07, 0.03, 0.05, 0.02, 0.03],
        "key": {"tonic": "E", "mode": "minor", "confidence": 0.51},
        "source": {"feature": "chroma_cqt", "hop_length": 512}
      },
      "timbre": {
        "mfcc_mean": [-101.3, 23.9, 17.7, 9.1, 1.1, -4.4, -6.8, -8.0, -5.2, -2.1, 0.7, 1.9, 1.2],
        "mfcc_std": [6.0, 3.6, 3.0, 2.6, 2.3, 2.1, 1.9, 1.7, 1.6, 1.4, 1.3, 1.2, 1.1],
        "spectral_flatness_mean": 0.19,
        "spectral_flatness_std": 0.04,
        "zero_crossing_rate_mean": 0.045,
        "zero_crossing_rate_std": 0.010
      },
      "spectral": {
        "centroid_mean": 2168.3,
        "centroid_std": 395.6,
        "bandwidth_mean": 1308.2,
        "bandwidth_std": 182.5,
        "rolloff_mean": 4879.8,
        "rolloff_std": 603.4
      },
      "energy": {}
    }
  },
  "debug": {
    "params": {
      "analysis_sr": 22050,
      "hop_length": 512,
      "phase_bins_per_bar": 16,
      "tempogram_peaks": 3,
      "energy_curve_points": 8
    },
    "fallbacks": [],
    "decisions": {
      "switch_in": {"window_start_s": 8.2, "window_end_s": 24.5, "quality": {"score": 1.42, "components": {"onset_clarity": 1.18, "tempo_stability": 0.91, "energy_ramp": 0.08, "spectral_flux": 0.34, "chroma_flux": 0.29}}},
      "switch_out": {"window_start_s": 176.4, "window_end_s": 193.2, "quality": {"score": 1.36, "components": {"onset_clarity": 1.09, "tempo_stability": 0.94, "energy_ramp": 0.04, "spectral_flux": 0.31, "chroma_flux": 0.33}}}
    },
    "errors": []
  }
}
```

Notes on selection and signature
- Switch windows are searched in intro/outro regions (first 30s and last 30s), aligned to beats when possible.
- Window scoring rewards stable tempo and clear percussive onsets and penalizes energy ramps, spectral flux, and chroma flux.
- Rhythmic signature combines tempogram peaks with a bar-phase onset pattern derived from percussive onsets.
- Phase patterns are aggregated across bars after testing tempo-scale hypotheses and beat offsets, then normalized for stability.
