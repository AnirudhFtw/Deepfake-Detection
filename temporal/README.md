# Temporal branch (not implemented yet)

This directory only holds the aggregator interface (`model.py`,
`TemporalAttentionPool`) so the spatial and frequency branches can be built
against a fixed contract: they need to expose a per-frame **embedding**
(the vector before the final classification layer — see `.embed()` on
`spatial/model.py` and `frequency/model.py`), not just a per-frame
real/fake logit.

`TemporalAttentionPool` learns to weight frames by how much evidence they
carry, replacing `predict_video.py`'s current `np.mean()` over per-frame
fake-probabilities with a learned signal.

## What's still needed here

See `tasks/IMPROVEMENT_PLAN.md`, Phase 4:

- A dataset that yields a *sequence* of per-video frame embeddings
  (from `spatial/model.py.embed()` and/or `frequency/model.py.embed()`),
  not one frame at a time.
- A training loop for `TemporalAttentionPool` (or a small transformer, if
  attention pooling underperforms) using that sequence dataset.

Deliberately not building the training pipeline yet — the fusion design in
Phase 5 will decide whether this trains standalone first or end-to-end
with the other branches, and building it twice would be wasted work.
