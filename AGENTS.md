Task: Update MixAnalyzer to improve performance by (1) removing full-track HPSS and running HPSS only inside switch windows, (2) using fast resampling on audio load, and (3) removing RMS/energy info from outputs only (keep RMS internally for window selection unchanged).

A) Performance refactor: remove full-track HPSS

In _analyze_sync, delete the full-track call:

y_harm, y_perc = librosa.effects.hpss(y)

Full-track feature computation must still happen, but without HPSS:

Keep computing onset_env_full / onset_times_full (can be based on raw y instead of y_perc).

Keep global beat tracking (beats_s_full) from the full-track onset envelope.

Keep full-track spectral/chroma flux computations (do not remove them; only adjust inputs if needed because y_harm/y_perc no longer exist).

Modify _analyze_section so HPSS happens only on the section audio:

Inside _analyze_section, after slicing y_section, compute:

y_harm_section, y_perc_section = librosa.effects.hpss(y_section)

Use y_perc_section for section onset/beat tracking and y_harm_section for harmony features, as before.

Update function signatures / call sites accordingly (remove passing y_harm/y_perc from full track).

B) Faster audio loading

Change the load call to:

librosa.load(abs_path, sr=self.analysis_sr, mono=True, res_type="kaiser_fast")

C) Output change only: remove RMS/energy fields from JSON

Important: Keep RMS analysis for window selection logic intact. Do not change _rms_curve usage in _score_window / _select_switch_window except to stop emitting values.

In the final analysis_json, remove RMS-derived output fields:

Remove track.overview.rms_mean (and any other RMS summary fields if present).

In per-section outputs (switch_in, switch_out):

Either omit "energy" entirely or set it to an empty dict {}.

Prefer {} if downstream consumers expect the key to exist.

In _empty_analysis, make the same decision consistently (omit or {}) so schema is stable.

Remove or keep _energy_features based on whether it is still used:

If it becomes unused after removing energy outputs, delete _energy_features and its call sites.

Do not delete _rms_curve because RMS is still used for window selection scoring.

D) Beat logic note

Keep both global and local beat tracking as currently implemented (including _choose_beats).

The global beat tracker no longer depends on HPSS; it should still exist for candidate generation and fallback.

_choose_beats should be modified so that it prefers the local beat tracking, than the global one. 

Deliverable: Updated code for MixAnalyzer that compiles and preserves existing analysis behavior, except:

HPSS no longer runs on the full track (only per switch window)

audio load uses kaiser_fast

RMS/energy is no longer included in outputs (but RMS scoring for window selection remains)