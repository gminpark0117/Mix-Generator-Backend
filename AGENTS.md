TASK: Implement deterministic MixRenderer (ordering + bar-aligned beatmatching)

A) Contracts + validation

Implement a real renderer in mix_renderer.py (keep PlaceholderMixRenderer intact).
Define “track identity” as TrackInput.mix_item_id and validate:
fixed_tracks is a subset of tracks (ignore/raise if not; choose one behavior and document).
No duplicates; preserve the exact order of fixed_tracks as the required prefix.
Validate analysis presence:
If any TrackInput.analysis_json is missing/invalid, fallback to deterministic “append order only” + simple crossfades.
B) Analysis extraction helpers (local-first)

Add helper functions to extract, for each track:
switch_in and switch_out blocks
rhythm: bpm_local, bpm_candidates, meter, downbeats_s, beats_s, signature.phase.pattern, signature.phase.stability, signature.periodicity.tempogram_top_peaks
harmony: key.tonic, key.mode, key.confidence
quality components: onset_clarity, tempo_stability, spectral_flux, chroma_flux, energy_ramp
C) Transition scoring (MOST IMPORTANT: tempo)

Implement compute_transition_options(A_out, B_in) -> list[TransitionOption]:
Evaluate tempo mapping using bpm_candidates (and optionally tempo_scale_used as a prior).
Pick best option by minimizing abs(delta_bpm_pct); hard-penalize if > ~6%.
Add secondary penalties:
key distance (weighted down if either key confidence is low) -- use music-theoretic notions, such as circle of fifths
signature mismatch (cosine similarity on phase.pattern; compare peak BPM sets from tempogram_top_peaks)
instability penalty using phase.stability.bar_to_bar_var
Determinism: all ties broken by (delta_bpm_pct, secondary_cost, mix_item_id).
D) Ordering with fixed prefix

Build final order:
prefix = fixed_tracks (exact order)
remaining = [t for t in tracks if t.mix_item_id not in prefix_ids]
Optimize order of remaining starting from the last prefix track:
Build pairwise transition costs using C).
Use an exact solver for small N (bitmask DP/Held–Karp up to e.g. 10), otherwise deterministic greedy.
Final order = prefix + optimized_remaining.
E) Build a deterministic “mix plan” (no audio yet)

For each adjacent pair (A->B), decide:
chosen tempo mapping (which BPM candidate on B)
target “mix BPM” (usually A.switch_out bpm)
overlap length in bars (longer when compatible; short when key/signature clash)
alignment anchors: downbeat-to-downbeat if available, else beat-to-beat fallback
F) Audio rendering (MVP)

Render SR: fixed (e.g. 44100), stereo float32 internal.
For each track:
load audio (soundfile or librosa), resample to render SR
select source regions based on switch windows (start near switch_in.start_s, end near switch_out.end_s)
For each transition A->B:
time-stretch B’s incoming region to target BPM (clamp stretch ratio)
align on bar boundaries using downbeats_s + meter.beats_per_bar (fallback to nearest beat)
equal-power crossfade over overlap region
“one bass at a time” MVP: apply simple HPF on incoming during early overlap, then release (scipy butter + lfilter)
G) Output + segments

Produce:
RenderResult.audio_bytes as WAV (16-bit PCM) deterministically
segments: list[SegmentSpec] with per-track start_ms/end_ms/source_start_ms (consistent even with overlaps)
Acceptance criteria (initial)

Respects fixed prefix exactly; deterministic order for remaining tracks.
Tempo compatibility dominates ordering decisions (local section BPMs).
Transitions align on downbeats when present; fall back cleanly when not.
Renderer is deterministic (no randomness; stable tie-breaks).
Produces valid WAV bytes + segments list for DB persistence.

