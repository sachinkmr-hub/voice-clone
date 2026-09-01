# VoiceGuard Python SDK

```bash
pip install -e sdk/python          # plus: pip install websockets  for the streaming client
```

## Analyse a recording

```python
from voiceguard_sdk import VoiceGuardClient

with VoiceGuardClient("http://localhost:8000", api_key="demo-key-sih26104") as client:
    result = client.analyze_file("call.wav", profile="wire_transfer",
                                 transaction_amount=4_200_000, claimed_role="CFO")
    print(result.explain())

    if result.inconclusive:
        request_callback()          # "we could not tell" is not "it was genuine"
    elif result.is_high_risk:
        escalate(result.top_factor)
```

## Gate a transaction

```python
approval = client.request_approval(result.session_id, amount=4_200_000)
if approval.blocked:
    raise TransactionBlocked(approval.reasons)
if approval.needs_step_up:
    require(approval.required_verification)
```

## Stream a live call

```python
with client.stream(profile="wire_transfer", identity="cfo_sharma") as stream:
    for chunk in telephony_bridge.pcm16_chunks():
        for risk in stream.send(chunk):
            if risk.is_high_risk:
                agent_ui.warn(risk.headline, risk.action)
    report = stream.finish()
```

The context manager guarantees the session closes even if your loop raises, so a crashed
integration cannot leak sessions on the server.

## Enrolment (turns on the cross-session check)

```python
for sample in ("cfo_1.wav", "cfo_2.wav", "cfo_3.wav"):
    client.enrol_file("cfo_sharma", sample)
```

Three or more samples are strongly preferred: with fewer, the within-speaker spread cannot
be estimated and the comparison falls back to a raw distance threshold.

## Notes

* A high risk score is **not** an exception. Only transport, auth and validation problems
  raise `VoiceGuardError`.
* `result.inconclusive` is the flag to check before treating a low score as a pass.
* Set `consent="recorded"` on the client if the deployment requires the consent header.
