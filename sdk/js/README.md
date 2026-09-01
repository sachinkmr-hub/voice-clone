# VoiceGuard JavaScript SDK

Zero dependencies. Works in browsers and Node 18+ (both have `fetch` and `WebSocket`).

## Browser: analyse an uploaded recording

```html
<script type="module">
  import { VoiceGuardClient } from './voiceguard.js';

  const client = new VoiceGuardClient({ baseUrl: 'http://localhost:8000' });

  document.querySelector('#file').addEventListener('change', async (event) => {
    const result = await client.analyzeFile(event.target.files[0], {
      profile: 'wire_transfer', transactionAmount: 4_200_000, claimedRole: 'CFO',
    });
    console.log(result.explain());
    if (result.inconclusive) requestCallback();
    else if (result.isHighRisk) showWarning(result.headline, result.action);
  });
</script>
```

For pages that cannot use modules, a plain `<script src="voiceguard.js">` exposes
`window.VoiceGuard`.

## Browser: live microphone

```js
const stream = client.stream({ profile: 'wire_transfer' });
stream.onRisk = (risk) => gauge.update(risk.score, risk.band);
stream.onAlert = (alert) => banner.show(alert.headline);
await stream.start();

const ctx = new AudioContext();
const source = ctx.createMediaStreamSource(await navigator.mediaDevices.getUserMedia({ audio: true }));
const node = ctx.createScriptProcessor(4096, 1, 1);
node.onaudioprocess = (e) => stream.send(toPcm16(e.inputBuffer.getChannelData(0), ctx.sampleRate));
source.connect(node);

// … later
const report = await stream.stop();
```

`toPcm16` averages over each source span when downsampling rather than decimating —
naive decimation folds aliases straight into the band the detector reads.

## Gate a transaction

```js
const approval = await client.requestApproval(result.sessionId, 4_200_000);
if (approval.blocked) throw new Error(approval.reasons.join(' '));
if (approval.needsStepUp) await requireSecondFactor(approval.required_verification);
```

## Notes

* A high risk score is a normal outcome, not an exception. Only transport, auth and
  validation problems throw `VoiceGuardError`.
* Check `result.inconclusive` before treating a low score as a pass.
