# Privacy & compliance note

## What VoiceGuard stores, by mode

| `RETENTION_MODE` | Audio on disk | Feature vectors | Scores & factors | Default |
|---|---|---|---|---|
| `none` | no | no | yes (score only) | |
| `features_only` | no | yes (float32, non-invertible) | yes | **✔ default** |
| `raw_audio` | yes, TTL-bounded | yes | yes | opt-in |

`features_only` keeps 80-dimensional summary statistics (means, variances, band energies,
pitch statistics). These are not sufficient to reconstruct intelligible speech and do not
constitute a voiceprint usable for enrolment elsewhere; they exist so that a disputed
decision can be re-examined without keeping the call.

## Identifiers

Every audit row carries `sha256(chunk_bytes)` and a random `session_id`. Phone numbers and
claimed identities are stored hashed with a per-deployment salt (`privacy/anonymize.py`)
unless `STORE_PII=true`.

## Retention enforcement

`privacy/retention.py` runs a periodic sweeper that deletes any record older than
`RETENTION_TTL_SECONDS` (default 86400) and any raw audio older than
`RAW_AUDIO_TTL_SECONDS` (default 900). Deletion is hard, not a soft flag.

## Consent and notification

The API rejects a stream that does not carry `X-Consent: recorded|analysed|none-required`
when `REQUIRE_CONSENT_HEADER=true`. The console displays a persistent banner naming the
active retention mode; do not remove it in a deployment.

## Edge / on-device path

`voiceguard.features` and `voiceguard.models` depend only on NumPy/SciPy and a small
scikit-learn model. They can be embedded in a handset or a PBX-side appliance so that only
the resulting score (a few hundred bytes) crosses the network. `scripts/edge_infer.py`
demonstrates this path with no server involved.

## Data-protection alignment

* Purpose limitation — features are used for authenticity scoring only.
* Storage limitation — TTL-enforced, default features-only.
* Explainability — every automated decision carries its contributing factors (relevant to
  automated-decision provisions in the DPDP Act 2023 and GDPR Art. 22 style regimes).
* Right to erasure — `DELETE /v1/sessions/{id}` removes all rows for a session.
