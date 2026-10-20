# Decisions

- TODO: `BaseAdapter.loss` has no negative-sample path. Disliked review samples are excluded from the next stage unless the adapter sets `supports_negatives` and implements `negative_loss`.
- TODO: an automatic decider can implement `ReviewDecider` and score samples before the next round; the human GUI and `lora review` are `HumanReviewDecider`, and this build has no automatic mode.
- The first review stage is `first_round_steps` (default 120, allowed 100–500). Later rounds split the remaining preset steps; the last round takes the remainder. `lora continue PROJECT --unattended` trains every remaining preset step in one stage and then finishes. `lora finish PROJECT` saves that LoRA, draws a final preview, and records `steps` and `steps_planned`. `lora finish PROJECT --discard` removes temporary checkpoints and samples and saves nothing.
