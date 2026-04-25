## Flow-Diagram

Base Model (frozen)
      +
SFT Adapter (trained in Phase 11)
      =
SFT Model → this becomes the starting point for DPO

Then DPO adds another adapter on top:

SFT Model (frozen)
      +
DPO Adapter (trained in Phase 15)
      =
DPO Model


## What DPO do and How ?

DPO works by comparing chosen vs rejected responses and adjusting probabilities:

```
For each preference pair:

Chosen  : "def reverse(s): return s[::-1]"  ← good answer
Rejected: "here is code... [cuts off]"       ← bad answer

DPO says:
→ Increase probability of generating chosen-style responses
→ Decrease probability of generating rejected-style responses
```

**Math in simple words:**

DPO computes a loss that:
- Rewards the model when it assigns **higher probability** to chosen response
- Penalizes the model when it assigns **higher probability** to rejected response

**Beta (0.1)** controls how strongly to push away from rejected responses. Lower beta = gentler adjustment.

After training:
- Model is more likely to generate complete, concise answers
- Model is less likely to generate verbose, cut-off answers

That is all DPO does — shift probabilities toward better responses.


# Where SFT and DPO Adapter place ?

Both SFT and DPO adapters follow same config — both placed on q_proj, k_proj, v_proj, o_proj only.