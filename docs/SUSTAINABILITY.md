# Economic and environmental sustainability

Both sections below are derived from **one measured number** — 124 ms p95 per 1-second
analysis window on a single CPU core (`python scripts/bench_latency.py`) — plus assumptions
that are labelled and shown, so a judge can substitute their own figures and redo the
arithmetic in their head.

---

## 1. Economic sustainability

### 1.1 Unit cost, from the measurement down

```
measured   p95 latency per window                 124 ms
            hop budget (a window every 0.5 s)      500 ms
            → concurrent calls per CPU core        500 / 124  ≈  4

[assumption] 8-vCPU cloud VM, on-demand           ₹30 / hour   (≈ US$0.35)
            → concurrent calls per VM              8 × 4 = 32
            → cost per concurrent-call-hour        ₹30 / 32  =  ₹0.94

[assumption] average monitored call length        3 minutes
            → calls per concurrent slot per hour   20
            → compute cost per call                ₹0.94 / 20  ≈  ₹0.047
```

**≈ 5 paise of compute per call monitored.** Add storage (feature vectors only, ~2 kB per
call, TTL 24 h — negligible) and bandwidth (audio already traverses the network; the score
is 252 bytes).

Sensitivity: even at 3× the VM price and half the assumed throughput, the figure is
₹0.28 per call. The conclusion — that compute is not the binding constraint on adoption —
holds across an order of magnitude of assumption error.

### 1.2 Why that number is small, and what it buys

The cost is low because **there is no GPU and no transformer in the inference path**. The
entire feature and model stack is NumPy/SciPy plus a 316 kB gradient-boosted model. A
wav2vec2-based detector of the kind most published work uses would need a GPU instance
(₹250–₹700/hour **[assumption]**) and would land 1–2 orders of magnitude higher per call.

That single architectural choice is what makes three things possible at once:

* deployment on commodity hardware a bank already owns, not a new GPU cluster;
* the on-device edge path (`scripts/edge_infer.py`), which is what unlocks the privacy
  story and the consumer/senior-citizen persona;
* a price point that survives contact with an Indian bank's procurement.

### 1.3 Where the money would come from

| Model | Buyer | Rough shape **[assumption]** |
|---|---|---|
| Per-seat SaaS | Bank / NBFC contact centre | Per agent seat per month; the value anchor is one prevented transfer, not the compute |
| Per-call API | Fintech, payments, KYC providers | Priced per call, at a large multiple of the ₹0.05 cost — the margin is in the model and the liability, not the CPU |
| On-prem licence | Large bank, government department | Annual licence; runs entirely inside their perimeter, which is often the only acceptable option |
| Telecom network layer | Operator | Per-subscriber per month, bundled as a spam/fraud-protection feature |
| Consumer app | Individuals, senior citizens | Freemium; the edge path means the free tier costs almost nothing to serve |

The honest framing: a single averted ₹42 lakh wire transfer pays for a very large number of
monitored calls. We are not in a position to claim a validated willingness-to-pay — we have
no pilot yet — and the roadmap in `docs/FEASIBILITY.md` puts a shadow-mode pilot first
precisely so the pricing conversation can be based on measured FPR rather than a slide.

### 1.4 Cost of *not* deploying

Voice-cloning fraud is asymmetric: the attacker's cost is a few seconds of scraped audio
and a free TTS account; the defender's loss is the full transfer. Any control that raises
the attacker's cost above "trivial" changes the economics. VoiceGuard does that at roughly
five paise a call.

---

## 2. Environmental sustainability

### 2.1 Energy and carbon, from the same measurement

```
[assumption] 8-vCPU VM under sustained load        ~50 W
            concurrent calls per VM (measured)      32
            average call length                     3 min = 0.05 h

            energy per concurrent-call-hour         50 W / 32 = 1.56 W
            energy per call                         1.56 W × 0.05 h = 0.078 Wh

[assumption] India grid intensity                   ~0.7 kg CO₂e / kWh
            → carbon per call                       0.078 Wh × 0.7 g/Wh ≈ 0.055 g CO₂e
```

**≈ 0.05 g CO₂e per call monitored.** For scale: roughly the footprint of one or two
web-page loads, and about four orders of magnitude below a single GPU-hour of model
training.

### 2.2 The design choices that produce it

This is not an accident of measurement; it is what the architecture was chosen for.

| Choice | Environmental consequence |
|---|---|
| **Hand-designed features + gradient boosting, not a transformer** | No GPU in the inference path at all. The dominant cost of most speech-AI systems is eliminated rather than optimised. |
| **Train once, infer many** | Training the shipped model takes ~2 minutes on one CPU core. There is no retraining loop, no continuous fine-tuning, no embedding-index rebuild. |
| **Silent windows are skipped** | `CallSession` does not score windows with no speech. On real calls, a substantial share of audio is silence, and none of it costs anything. |
| **Edge inference** | Moves computation to a device that is already powered on and removes the audio uplink entirely. The marginal datacentre energy for an edge-scored call is zero. |
| **CPU-only container** | The backend image needs no CUDA layer; it is a `python:3.11-slim` base. Smaller images mean less registry storage, less network transfer on every deploy, and faster cold starts. |
| **Bounded retention** | Feature vectors are hard-deleted after 24 h by default. No indefinitely-growing storage tier quietly drawing power for years. |

### 2.3 What we are not claiming

We have not run a life-cycle assessment, we have not metered actual wall power (the 50 W
figure is an assumption, not a measurement), and the grid-intensity figure is a national
average that varies enormously by region and time of day. The claim we will defend is the
*relative* one, which does not depend on those numbers being precise:

> A CPU-only classical-feature detector uses one to two orders of magnitude less energy
> per call than a GPU-served neural equivalent, and this repository demonstrates that the
> CPU-only version is fast enough to meet the latency requirement with 4× headroom.

### 2.4 Second-order effect

The largest environmental term in fraud is not the detection compute — it is the fraud
itself: the investigations, the call-backs, the re-issued instruments, the travel, the
court time. Preventing a transfer at the point of the call avoids all of that downstream
activity. We cannot quantify it credibly, so we mention it once and do not build a claim
on it.
