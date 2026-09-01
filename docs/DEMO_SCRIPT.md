# 6-minute demo script

**0:00 — The setup.** "This is a bank's approval desk. A call comes in from someone who
sounds exactly like the CFO, asking to release a ₹42 lakh transfer."

**0:30 — Start the stack.** `make run` → open `/console`. Point out: nothing is recorded,
the retention banner reads *features-only*.

**1:00 — Genuine call.** Stream `data/corpus/bonafide_*.wav` (or speak into the mic).
The gauge settles in the low 20s, band `LOW`, factors show "natural pitch micro-variation",
"broadband energy to 8 kHz". Latency counter shows ~0.5 s per update.

**2:00 — Cloned call.** Stream `data/corpus/synthetic_*.wav`. Within two windows the gauge
crosses 60 → `HIGH`, then `CRITICAL`. Read the factors aloud — this is the differentiator:
*"high-frequency energy cliff at 7.6 kHz"*, *"pitch micro-variation 3.8σ below human range"*,
*"digital-silence noise floor"*.

**3:00 — The alert.** The alert panel fires; show the webhook payload in the log pane and
the SMS/email channel stubs.

**3:30 — The bank integration.** Switch to the "Approval" tab. Submit the ₹42 lakh transfer
with the live `session_id`: the mock core-banking endpoint returns `block` with the reason
chain. Re-submit with the genuine session: `allow`.

**4:30 — Cross-session check.** Enrol the genuine speaker, replay the clone → L3 lights up
with "speaker embedding distance 0.62 vs enrolment".

**5:00 — Policy.** Admin tab: drop the CRITICAL threshold for the "wire transfer" profile
from 80 to 65, replay — the same audio now escalates earlier. Policy is separable from the
model.

**5:30 — Honesty slide.** `docs/MODEL_CARD.md`: what it was trained on, measured EER, and
the three failure modes we know about.
