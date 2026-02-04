# TASK: Refactor MixAnalyzer to librosa-only + local-first descriptors + new rhythmic signature

You previously generated `mix_analyzer.py` containing a concrete `MixAnalyzer` class and an `AnalysisResult` dataclass. The current analyzer produces global rhythm/harmony/timbre/spectral features and uses BeatNet optionally.

Major changes required:
1) **Remove BeatNet entirely** (no BeatNet/madmom/pyaudio dependencies).
2) Make analysis **local-first**: compute *most* descriptors on the **switch_in** and **switch_out** sections (because those are what matter for beatmatching).
3) Replace the old groove vector approach with a more robust **rhythmic signature**: (A) periodicity peaks from a tempogram + (B) bar-phase accent pattern computed from percussive onset envelope.
4) Produce a markdown schema doc with field summaries + an example `analysis_json`.

## Non-negotiable contract
- `async def analyze(self, abs_path: str, metadata_json: dict) -> AnalysisResult`
- Return `metadata_json` **exactly as input** (do not modify).
- Output must be JSON-serializable (no numpy arrays).

---

## Critique of current implementation (what to fix)

### A) Dependency complexity + BeatNet parsing heuristics
- Delete all BeatNet-related code: imports, init, debug fields, parsing logic.
- Use librosa-only for beat tracking.

### B) Rhythm features must be drum-forward
- Use HPSS and compute onset-related features on the **percussive** component:
  - `y_harm, y_perc = librosa.effects.hpss(y)`
  - `onset_env = librosa.onset.onset_strength(y=y_perc, ...)`
- Use percussive onset features for:
  - selecting switch windows
  - rhythmic signature computation
  - section tempo stability scoring

### C) Switch selection must prefer “mixable” regions, not busy drops
Old scoring over-rewards busy high-onset regions.
Improve scoring by:
- rewarding: stable tempo, clear percussive onsets, steady energy
- penalizing: strong energy ramp (build), high spectral flux, high chroma flux (proxy for vocals/harmonic churn)
- aligning windows to bar boundaries once meter/phase is inferred (or at least to beat boundaries as a fallback)

### D) Beat subdivision ambiguity (beat vs tatum)
Librosa may detect sub-beats as beats in some tracks.
Mitigate by evaluating multiple tempo-scale hypotheses in rhythm signature / meter inference:
- tempo_scale ∈ {0.5, 1.0, 2.0}
Choose the most stable hypothesis per section.

---

## New output design: local-first

### Top-level required fields
`analysis_json` MUST contain:
- `schema_version: 2` (increment because schema changed materially)
- `track`: {
    `duration_s`, `sr`, `hop_length`,
    optional minimal `overview` fields (keep light)
  }
- `sections.switch_in`
- `sections.switch_out`
- `debug` (params, fallbacks, decisions)

### Local section blocks (switch_in / switch_out)
Each of `sections.switch_in` and `sections.switch_out` MUST contain:
- `start_s`, `end_s`
- `quality`: numeric score + diagnostic sub-scores (stability, energy_ramp, flux, etc.)
- `rhythm`: {
    `beats_s` (local beat times, seconds),
    `bpm_local`,
    `bpm_candidates` (half/double-time),
    `meter`: {beats_per_bar, note_value=4, confidence, assumed},
    `downbeats_s`,
    `beat_positions` (1..beats_per_bar, same length as beats_s),
    `beat_interval_s` stats (mean/std/cv/bpm_std),
    `signature` (defined below)
  }
- `harmony` (computed on THIS section): chroma summary + key estimate 
- `timbre` (computed on THIS section): MFCC summary etc. 
- `spectral` 
- `energy` (optional): rms stats + small downsampled curve within the section

### Rhythmic signature (replaces old groove)
Remove old `groove.steps_per_bar` and do NOT rely on `steps_per_beat` being stable.
Instead implement:

`rhythm.signature` = {
  `onset_source`: "hpss_percussive",
  `periodicity`: {
     `tempogram_top_peaks`: list of {bpm, strength} (K≈5),
     optional `tempogram_profile_ds`: downsampled/compact vector
  },
  `phase`: {
     `phase_bins_per_bar`: integer (e.g., 96),
     `pattern`: length B list[float] (normalized),
     `fft_mag`: first M magnitudes (e.g., 16) (optional but useful),
     `stability`: {bar_to_bar_var, n_bars_used},
     `tempo_scale_used`: 0.5/1.0/2.0,
     `bar_offset_used`: inferred offset index
  }
}

How to compute `phase.pattern`:
- Use inferred meter + bar phase within the SECTION.
- Split into bars (downbeat->downbeat).
- For each bar, resample the percussive onset envelope across that bar into B bins.
- Aggregate across bars with median/mean to get a stable pattern.
- Normalize (L1 or max) robustly.

---

## Implementation plan (two-pass recommended)

### Pass 1: load audio + compute cheap global primitives
- Load audio mono at `analysis_sr`.
- Compute HPSS percussive `y_perc`.
- Compute percussive onset envelope `onset_env` and its timestamps.
- Get an initial beat grid for the full track:
  - `tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, ...)`
  - `beats_s = frames_to_time(beat_frames, ...)`
This full-track beat grid is used only for candidate windowing and as a fallback.

### Pass 2: select switch_in/out windows
- Search within intro region (first min(30% duration, 60s)) for switch_in candidates.
- Search within outro region (last min(30% duration, 60s)) for switch_out candidates.
- Candidate windows should be bar/beat aligned if possible; otherwise beat aligned.
- Score windows with mix-aware objective:
  - + percussive onset clarity
  - + tempo stability (low IBI CV)
  - - energy slope (RMS ramp)
  - - spectral flux / chroma flux
Pick the best window for each.

### Pass 3: compute section-local descriptors (the important part)
For each chosen section:
- Slice audio `y_section`, and compute percussive onset env for that slice (or reuse global via time slicing).
- Compute a section-local beat grid:
  - either run beat_track on the slice, OR subset global beats that fall in [start,end]
  - prefer whichever yields more stable intervals (choose by CV)
- Infer meter + bar phase within the section:
  - try beats_per_bar candidates [4,3]
  - try tempo_scale hypotheses {0.5,1.0,2.0}
  - choose the best by downbeat onset score + stability (bar_to_bar_var)
- Compute `rhythm.signature` (tempogram peaks + phase pattern).
- Compute harmony/timbre/spectral/energy ON THE SECTION (not full track).

### Minimal global fields in `track`
Even though we’re local-first, include:
- duration, sr, hop_length
Optionally include a very light overview:
- `overview.global_bpm_guess` (from full-track beats) and/or `overview.rms_mean`
But do NOT compute heavy features globally unless needed for robustness.

---

## Robustness / fallbacks
- If beat tracking yields too few beats in a section:
  - fall back to full-track beats subset
  - if still insufficient, return empty section descriptors with clear `debug.fallbacks` entries
- Ensure all arrays are converted to lists of python floats/ints.

---

## Required documentation output (NEW)
In addition to code changes:
1) Create a markdown file: `docs/analysis_json_schema.md`
2) This file MUST include:
   - A short description of analyzer goals
   - A bullet list summary of every field produced in `analysis_json` (including nested fields)
   - An **example** `analysis_json` (realistic-looking numbers, not placeholders like "foo"), formatted as JSON code block
   - Notes on how switch windows are selected and how rhythmic signature is computed

---

## Acceptance criteria
- No BeatNet/madmom/pyaudio imports.
- `analysis_json` is local-first: tempo/chroma/mfcc/spectral are computed primarily on switch_in/out sections.
- Rhythmic signature uses tempogram peaks + bar-phase pattern (no steps_per_bar; no reliance on fixed steps_per_beat).
- Output is JSON-serializable and stable for same input.
- `docs/analysis_json_schema.md` is created and accurate.
- `metadata_json` is unchanged.
