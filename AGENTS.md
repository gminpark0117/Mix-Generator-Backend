## Refactor Plan: Add Variable-BPM MixRenderer Variant (tempo ramps during transitions)

### Goal
Add a new renderer variant that keeps each track at its **native BPM outside transitions**, and performs a **tempo ramp only during the transition window**:
- For `A -> B`, mix tempo stays at `bpm(A)` before transition.
- During overlap, tempo ramps linearly from `bpm(A)` to `bpm(B)`.
- After overlap, mix tempo is `bpm(B)`.

Keep existing `DeterministicMixRenderer` behavior untouched; the new variant is opt-in (pipeline tester switch).

---

### A) New renderer class + selection plumbing
- Add `VariableBpmMixRenderer` in `atomix/renderers/mix_renderer.py` implementing the same `render(tracks, fixed_tracks)` contract.
- Keep `PlaceholderMixRenderer` unchanged.
- Update `mix_pipeline_tester.py` to accept `--renderer {deterministic,variable_bpm}` (default `deterministic`) and instantiate the chosen renderer.
- Include `renderer_variant` and any new debug fields in the tester’s JSON output for easy comparison.

---

### B) Stretching logic (tempo ramp during overlap for BOTH tracks)
For each transition `A -> B`, both the outgoing overlap of `A` and the incoming overlap of `B`
must be time-warped so they share the *same instantaneous BPM* throughout the overlap.

- Define the shared tempo curve over the overlap `t∈[0,T]`:
  - `bpm(t) = bpm_a + (bpm_b - bpm_a) * (t/T)` (linear ramp)
- Approximate `bpm(t)` deterministically with `K` chunks (e.g. 16/32):
  - Choose per-chunk target tempo `bpm_k` (use endpoints for k=0/k=K-1 to ensure continuity):
    - `bpm_0 = bpm_a`, `bpm_{K-1} = bpm_b`
- Warp outgoing overlap (A) and incoming overlap (B) to the SAME `bpm_k` per chunk:
  - For A: `rate_out_k = bpm_k / bpm_a`
  - For B: `rate_in_k  = bpm_k / bpm_b`
  - Apply `librosa.effects.time_stretch(..., rate_*)` on each chunk, trim/pad so each chunk outputs
    exactly `overlap_n/K` samples (so both warped overlaps have exactly `overlap_n` samples total).
- Crossfade the two warped overlaps (equal-power), then append:
  - `A_before` un-stretched (native bpm_a)
  - `B_after` un-stretched (native bpm_b)

Continuity:
- Because `rate_out_0 = 1` and `rate_in_{K-1} = 1`, boundaries between un-stretched audio and warped overlap
  are continuous in tempo (only the overlap is affected).
---

### C) Transition scoring update (raw BPM delta + feasibility for BOTH sides)
When evaluating `A -> B` options (choosing which `bpm_b` candidate to use):
- Tempo delta term uses RAW BPM change:
  - `delta_raw_pct = abs(bpm_b - bpm_a) / bpm_a * 100`
- Feasibility/hard penalty must consider BOTH tracks’ required endpoint stretch:
  - Outgoing end-rate: `end_rate_out = bpm_b / bpm_a`
  - Incoming start-rate: `start_rate_in = bpm_a / bpm_b`
  - If either is outside `[MIN_STRETCH_RATE, MAX_STRETCH_RATE]`, treat option as invalid/huge cost
    (otherwise A and B cannot “agree on BPM always” during overlap under clamp).
- Secondary penalties unchanged (key/signature/instability), determinism tie-break unchanged.
---

### D) Ordering (DP/greedy) uses the variable-BPM transition costs
- In `VariableBpmMixRenderer._order_tracks`, build `cost_map[(A,B)]` using the **variable-BPM** transition option’s `total_cost`.
- Keep the existing Held–Karp DP for small N and deterministic greedy for large N.
- Fixed prefix handling remains identical.

---

### E) Debug + acceptance checks
Add debug fields to the variable renderer:
- per transition: `bpm_a`, chosen `bpm_b`, `delta_raw_pct`, `start_rate`, `start_rate_clamped`, `start_match_error_pct`, `K` chunk count.
- assert determinism: no randomness, stable sorting, consistent rounding.

Acceptance:
- Outside overlap windows, tracks play at native tempo (no full-chunk time-stretch).
- During overlap, incoming tempo ramps from `bpm_a` to `bpm_b`.
- Transition scoring minimizes **raw** BPM change magnitude.
- Pipeline tester can switch renderer variants and produce comparable debug output.