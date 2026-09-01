/* VoiceGuard console — vanilla JS, no build step.
 *
 * Three input paths share one rendering layer, deliberately: an uploaded recording and a
 * live microphone stream go through the same server pipeline and are drawn by the same
 * code, so what a judge sees in the file demo is exactly what happens on a live call.
 */
'use strict';

const API = '';
const BANDS = ['LOW', 'ELEVATED', 'HIGH', 'CRITICAL'];
const BAND_COLOR = { LOW: '#059669', ELEVATED: '#d97706', HIGH: '#ea580c', CRITICAL: '#dc2626' };
const ARC_LENGTH = 251.3;   // path length of the gauge arc

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, options) {
  const response = await fetch(API + path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

/* ─────────────────────────────────────────────────────────────── tabs ── */

document.querySelectorAll('nav.tabs button').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('nav.tabs button').forEach((b) =>
      b.setAttribute('aria-selected', String(b === button)));
    document.querySelectorAll('.panel').forEach((p) =>
      p.classList.toggle('active', p.id === 'panel-' + button.dataset.tab));
    if (button.dataset.tab === 'approval') refreshSessions();
    if (button.dataset.tab === 'policy') { loadProfiles(); loadEnrolments(); loadSystem(); }
  });
});

/* ────────────────────────────────────────────────────────────── gauge ── */

function paintGauge(arcId, valueId, bandId, score, band) {
  const arc = $(arcId);
  const clamped = Math.max(0, Math.min(100, Number(score) || 0));
  arc.style.strokeDashoffset = String(ARC_LENGTH * (1 - clamped / 100));
  arc.setAttribute('stroke', BAND_COLOR[band] || BAND_COLOR.LOW);
  $(valueId).textContent = Math.round(clamped);
  $(valueId).setAttribute('fill', BAND_COLOR[band] || '#131a35');
  const label = $(bandId);
  label.textContent = band;
  label.className = 'band-label band-' + band;
}

function paintTimeline(svgId, points) {
  const svg = $(svgId);
  if (!points.length) { svg.innerHTML = ''; return; }
  const W = 600, H = 118, PAD = 6;
  const step = points.length > 1 ? (W - PAD * 2) / (points.length - 1) : 0;
  const y = (v) => H - PAD - (Math.max(0, Math.min(100, v)) / 100) * (H - PAD * 2);

  const line = points.map((p, i) => `${PAD + i * step},${y(p.score)}`).join(' ');
  const area = `${PAD},${H - PAD} ${line} ${PAD + (points.length - 1) * step},${H - PAD}`;
  const last = points[points.length - 1];

  // Threshold guides make the curve readable without a legend.
  const guides = [[35, '#d97706'], [60, '#ea580c'], [80, '#dc2626']].map(
    ([v, c]) => `<line x1="${PAD}" y1="${y(v)}" x2="${W - PAD}" y2="${y(v)}"
       stroke="${c}" stroke-width="1" stroke-dasharray="3 4" opacity=".28"/>`).join('');

  svg.innerHTML = `
    <defs><linearGradient id="tl-${svgId}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${BAND_COLOR[last.band] || '#6366f1'}" stop-opacity=".30"/>
      <stop offset="100%" stop-color="${BAND_COLOR[last.band] || '#6366f1'}" stop-opacity="0"/>
    </linearGradient></defs>
    ${guides}
    <polygon points="${area}" fill="url(#tl-${svgId})"/>
    <polyline points="${line}" fill="none" stroke="${BAND_COLOR[last.band] || '#6366f1'}"
              stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${PAD + (points.length - 1) * step}" cy="${y(last.score)}" r="4"
            fill="${BAND_COLOR[last.band] || '#6366f1'}"/>`;
}

function renderFactors(target, factors, counter) {
  const container = $(target);
  const all = (factors || []).map((f) => ({ ...f, genuine: f.direction === 'genuine' }))
    .concat((counter || []).map((f) => ({ ...f, genuine: true })));
  if (!all.length) {
    container.innerHTML = '<div class="empty" style="padding:20px">' +
      'No single factor stood out — the score reflects the overall picture.</div>';
    return;
  }
  container.innerHTML = all.map((f) => `
    <div class="factor ${f.genuine ? 'genuine' : ''}">
      <div class="factor-head">
        <span class="factor-label">${f.layer ? `<span class="layer-tag">${esc(f.layer)}</span>` : ''}${esc(f.label)}</span>
        <span class="factor-pct">${Math.round((f.contribution || 0) * 100)}%</span>
      </div>
      ${f.detail ? `<div class="factor-detail">${esc(f.detail)}</div>` : ''}
      <div class="factor-bar"><div class="factor-fill" style="width:${Math.min(100, (f.contribution || 0) * 100)}%"></div></div>
    </div>`).join('');
}

function renderLayers(layers) {
  const card = $('layers-card');
  if (!layers || !layers.length) { card.hidden = true; return; }
  card.hidden = false;
  $('layers').innerHTML = layers.map((l) => `
    <div class="layer-row">
      <div>
        <div class="layer-name">${esc(l.label || l.layer)}
          <span class="status-tag status-${esc(l.status)}">${esc(l.status)}</span></div>
        ${l.reason ? `<div class="layer-reason">${esc(l.reason)}</div>` : ''}
      </div>
      <div class="layer-score">
        ${l.status === 'voted' ? Math.round(l.score * 100) + '/100' : '—'}
        <div style="font-size:11px;color:var(--ink-faint);font-weight:400">
          ${l.status === 'voted' ? Math.round((l.weight_share || 0) * 100) + '% of decision' : 'no vote'}</div>
      </div>
    </div>`).join('');
}

function renderCaveats(caveats) {
  const box = $('caveats');
  if (!caveats || !caveats.length) { box.innerHTML = ''; return; }
  box.innerHTML = `<div class="caveats"><strong>Read this alongside the score:</strong>
    <ul>${caveats.map((c) => `<li>${esc(c)}</li>`).join('')}</ul></div>`;
}

/* ──────────────────────────────────────────────────────────── analyse ── */

let lastSessionId = null;

function collectContext() {
  const data = new FormData();
  data.append('language', $('ctx-language').value);
  data.append('profile', $('ctx-profile').value);
  const amount = Number($('ctx-amount').value) || 0;
  if (amount > 0) data.append('transaction_amount', String(amount));
  if ($('ctx-known').checked) data.append('known_contact', 'true');
  if ($('ctx-verified').checked) data.append('caller_id_verified', 'true');
  if ($('ctx-role').value.trim()) data.append('claimed_role', $('ctx-role').value.trim());
  if ($('ctx-transcript').value.trim()) data.append('transcript', $('ctx-transcript').value.trim());
  return data;
}

async function analyzeBlob(blob, filename) {
  const zone = $('dropzone');
  const original = zone.innerHTML;
  zone.innerHTML = '<div class="icon"><span class="spinner" style="border-color:rgba(99,102,241,.3);border-top-color:#6366f1"></span></div>' +
                   `<strong>Analysing ${esc(filename)}…</strong><span>Running the streaming pipeline</span>`;
  try {
    const data = collectContext();
    data.append('file', blob, filename);
    const result = await api('/v1/analyze/file', { method: 'POST', body: data });
    showResult(result);
    lastSessionId = result.session_id;
    refreshSessions();
  } catch (error) {
    zone.innerHTML = `<div class="icon">⚠️</div><strong>${esc(error.message)}</strong>
                      <span>Click to try another file</span>`;
    setTimeout(() => { zone.innerHTML = original; }, 4200);
    return;
  }
  zone.innerHTML = original;
}

function showResult(result) {
  $('result-empty').hidden = true;
  $('result').hidden = false;
  paintGauge('gauge-arc', 'score-value', 'band-label', result.score, result.band);
  $('headline').textContent = result.headline || '';
  const action = $('action-box');
  action.textContent = result.action || '';
  action.className = 'action-box b-' + result.band;

  if (result.timeline && result.timeline.length) {
    $('timeline-card').hidden = false;
    paintTimeline('timeline', result.timeline.map((t) => ({ score: t.score, band: t.band })));
    $('stat-grid').innerHTML = [
      ['Verdict', String(result.verdict || '').replace(/_/g, ' ')],
      ['Peak', Math.round(result.peak_score) + '/100'],
      ['Duration', (result.duration_seconds || 0).toFixed(1) + 's'],
      ['Windows', result.windows_analyzed],
      ['Latency', Math.round(result.mean_latency_ms) + 'ms'],
    ].map(([k, v]) => `<div class="stat"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join('');
  } else {
    $('timeline-card').hidden = true;
  }

  $('factors-card').hidden = false;
  renderFactors('factors', result.factors, null);
  renderCaveats(result.caveats);
  renderLayers(result.layers);
  $('result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* dropzone wiring */
const dropzone = $('dropzone');
dropzone.addEventListener('click', () => $('file-input').click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file-input').click(); }
});
['dragenter', 'dragover'].forEach((event) =>
  dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.add('over'); }));
['dragleave', 'drop'].forEach((event) =>
  dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.remove('over'); }));
dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) analyzeBlob(file, file.name);
});
$('file-input').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) analyzeBlob(file, file.name);
  e.target.value = '';
});

/* ───────────────────────────────────────────────────────── scenarios ── */

async function loadScenarios() {
  try {
    const data = await api('/v1/demo/scenarios');
    $('scenario-chips').innerHTML = data.scenarios.map((s, i) => `
      <button class="chip ${s.kind === 'bonafide' ? 'genuine' : 'clone'}" data-scenario="${i}"
              title="${esc(s.story)}">${s.kind === 'bonafide' ? '✅' : '🤖'} ${esc(s.title)}</button>`).join('');
    document.querySelectorAll('[data-scenario]').forEach((button) => {
      button.addEventListener('click', () => runScenario(data.scenarios[Number(button.dataset.scenario)]));
    });
  } catch (error) {
    $('scenario-chips').innerHTML = `<span class="empty" style="padding:8px">${esc(error.message)}</span>`;
  }
}

async function runScenario(scenario) {
  const context = scenario.call_context || {};
  $('ctx-amount').value = context.transaction_amount || 0;
  $('ctx-language').value = scenario.language || 'auto';
  $('ctx-role').value = context.claimed_role || '';
  $('ctx-transcript').value = context.transcript || '';
  $('ctx-known').checked = context.known_contact === true;
  $('ctx-verified').checked = context.caller_id_verified === true;
  if (context.transaction_amount > 500000) $('ctx-profile').value = 'wire_transfer';

  const params = new URLSearchParams({
    kind: scenario.kind, seconds: '7', speaker: String(scenario.speaker ?? 1),
    language: scenario.language || 'hi-IN',
  });
  if (scenario.method) params.set('method', scenario.method);

  const response = await fetch(`/v1/demo/sample?${params}`);
  const blob = await response.blob();
  analyzeBlob(blob, `${scenario.id}.wav`);
}

/* ───────────────────────────────────────────────────────── live call ── */

let audioContext = null, mediaStream = null, processor = null, socket = null;
let liveTimeline = [], liveStart = 0, liveTimer = null;

function log(message, kind) {
  const box = $('live-log');
  const time = new Date().toLocaleTimeString();
  box.innerHTML += `<div><span class="t">${time}</span> <span class="${kind || ''}">${esc(message)}</span></div>`;
  box.scrollTop = box.scrollHeight;
}

/** Downsample the browser's native rate (usually 48 kHz) to the 16 kHz the API expects.
 *  Averaging over each source span rather than picking one sample avoids the aliasing a
 *  naive decimation would fold straight into the band our detector reads. */
function downsampleTo16k(input, inputRate) {
  if (inputRate === 16000) return input;
  const ratio = inputRate / 16000;
  const output = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < output.length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    output[i] = end > start ? sum / (end - start) : 0;
  }
  return output;
}

function floatToPcm16(input) {
  const buffer = new ArrayBuffer(input.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

async function startMic() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false,
               autoGainControl: false },
    });
  } catch (error) {
    log('Microphone permission denied: ' + error.message, 'err');
    return;
  }

  $('mic-start').disabled = true;
  $('mic-stop').disabled = false;
  liveTimeline = [];
  $('live-log').innerHTML = '';
  $('live-factors').innerHTML = '<div class="empty" style="padding:22px">Listening…</div>';

  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${protocol}://${location.host}/v1/stream`);
  socket.binaryType = 'arraybuffer';

  socket.onopen = () => {
    log('WebSocket open → sending start frame', 'ok');
    socket.send(JSON.stringify({
      action: 'start', profile: $('ctx-profile').value || 'default',
      language: $('ctx-language').value, encoding: 'pcm16', sample_rate: 16000,
    }));
    liveStart = Date.now();
    liveTimer = setInterval(() => {
      $('live-elapsed').textContent = Math.round((Date.now() - liveStart) / 1000) + 's';
    }, 500);
  };

  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'started') {
      log(`Session ${message.session_id} · window ${message.window_seconds}s / hop ${message.hop_seconds}s`, 'ok');
      lastSessionId = message.session_id;
    } else if (message.type === 'risk') {
      liveTimeline.push({ score: message.score, band: message.band });
      if (liveTimeline.length > 120) liveTimeline.shift();
      paintGauge('live-arc', 'live-score', 'live-band', message.score, message.band);
      paintTimeline('live-timeline', liveTimeline);
      $('live-headline').textContent = message.headline || '';
      const action = $('live-action');
      action.hidden = false;
      action.textContent = message.action || '';
      action.className = 'action-box b-' + message.band;
      $('live-windows').textContent = liveTimeline.length;
      $('live-latency').textContent = Math.round(message.latency_ms) + 'ms';
      renderFactors('live-factors', message.factors, null);
    } else if (message.type === 'alert') {
      log(`ALERT ${message.band} — ${message.headline}`, 'warn');
    } else if (message.type === 'final') {
      log(`Final verdict: ${message.report.verdict} (peak ${message.report.peak_score})`, 'ok');
      refreshSessions();
    } else if (message.type === 'error') {
      log('Error: ' + message.error, 'err');
    }
  };

  socket.onerror = () => log('WebSocket error', 'err');
  socket.onclose = () => log('WebSocket closed');

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioContext.createMediaStreamSource(mediaStream);
  // ScriptProcessorNode is deprecated but is the only path that works without shipping a
  // separate worklet file, which would break the "one HTML file, no build" promise.
  processor = audioContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    let peak = 0;
    for (let i = 0; i < input.length; i++) peak = Math.max(peak, Math.abs(input[i]));
    $('input-meter').style.width = Math.min(100, peak * 140) + '%';
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(floatToPcm16(downsampleTo16k(input, audioContext.sampleRate)));
    }
  };

  source.connect(processor);
  // Route to a muted gain node: some browsers will not run a ScriptProcessor that is not
  // connected to the destination, but connecting it directly would echo the caller.
  const mute = audioContext.createGain();
  mute.gain.value = 0;
  processor.connect(mute);
  mute.connect(audioContext.destination);
  log(`Capturing at ${audioContext.sampleRate} Hz → downsampling to 16000 Hz`);
}

function stopMic() {
  $('mic-start').disabled = false;
  $('mic-stop').disabled = true;
  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ action: 'stop' }));
  }
  if (processor) { processor.disconnect(); processor.onaudioprocess = null; processor = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
  $('input-meter').style.width = '0%';
  log('Stopped');
}

$('mic-start').addEventListener('click', startMic);
$('mic-stop').addEventListener('click', stopMic);
window.addEventListener('beforeunload', () => { if (socket) socket.close(); });

/* ──────────────────────────────────────────────────────────── approval ── */

async function refreshSessions() {
  try {
    const sessions = await api('/v1/sessions?include_closed=true&limit=25');
    const select = $('apr-session');
    const current = select.value || lastSessionId;
    select.innerHTML = sessions.length
      ? sessions.map((s) => `<option value="${esc(s.session_id)}">
          ${esc(s.session_id)} — ${Math.round(s.score)}/100 ${esc(s.band)}</option>`).join('')
      : '<option value="">— analyse a call first —</option>';
    if (current && sessions.some((s) => s.session_id === current)) select.value = current;
  } catch (_) { /* the selector is a convenience; never block the page on it */ }
}

$('apr-submit').addEventListener('click', async () => {
  const sessionId = $('apr-session').value;
  if (!sessionId) {
    $('apr-result').innerHTML = '<div class="empty" style="padding:24px">' +
      'Analyse a recording first — approval is gated on a live voice session.</div>';
    return;
  }
  $('apr-submit').disabled = true;
  try {
    const result = await api('/v1/integrations/bank/approval', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        amount: Number($('apr-amount').value) || 0,
        profile: $('apr-profile').value,
        beneficiary: $('apr-beneficiary').value,
        reference: $('apr-reference').value,
      }),
    });
    const icon = { allow: '✅', step_up: '🔐', block: '⛔' }[result.decision] || '❓';
    $('apr-result').innerHTML = `
      <div class="decision ${esc(result.decision)}">
        <div class="verdict">${icon} ${esc(result.decision.replace('_', ' '))}</div>
        <div class="msg">${esc(result.message)}</div>
      </div>
      <div style="font-size:12.5px;color:var(--ink-faint);margin-bottom:9px">
        Voice risk <strong>${result.risk_score}/100</strong> (${esc(result.band)}) ·
        ref <code>${esc(result.reference)}</code></div>
      <ul class="reasons">${result.reasons.map((r) => `<li>${esc(r)}</li>`).join('')}</ul>
      ${result.required_verification.length ? `<div style="margin-top:11px;font-size:12.5px">
        <strong>Required before proceeding:</strong> ${result.required_verification.map(esc).join(', ')}</div>` : ''}`;
  } catch (error) {
    $('apr-result').innerHTML = `<div class="empty" style="padding:24px">⚠️ ${esc(error.message)}</div>`;
  } finally {
    $('apr-submit').disabled = false;
  }
});

/* ────────────────────────────────────────────────────────────── policy ── */

async function loadProfiles() {
  try {
    const profiles = await api('/v1/admin/profiles');
    const body = $('profiles-table').querySelector('tbody');
    body.innerHTML = profiles.map((p) => `
      <tr data-profile="${esc(p.name)}">
        <td><strong>${esc(p.name)}</strong><div style="font-size:11.5px;color:var(--ink-faint)">${esc(p.description)}</div></td>
        <td><input type="number" value="${p.elevated}" data-k="elevated" style="width:64px"></td>
        <td><input type="number" value="${p.high}" data-k="high" style="width:64px"></td>
        <td><input type="number" value="${p.critical}" data-k="critical" style="width:64px"></td>
        <td><button class="btn ghost small" data-save="${esc(p.name)}">Save</button></td>
      </tr>`).join('');

    ['ctx-profile', 'apr-profile'].forEach((id) => {
      const select = $(id);
      const current = select.value;
      select.innerHTML = profiles.map((p) => `<option value="${esc(p.name)}">${esc(p.name)}</option>`).join('');
      select.value = profiles.some((p) => p.name === current) ? current
        : (id === 'apr-profile' ? 'wire_transfer' : 'default');
    });

    body.querySelectorAll('[data-save]').forEach((button) => {
      button.addEventListener('click', async () => {
        const row = button.closest('tr');
        const payload = { name: button.dataset.save, description: '', alert_channels: ['websocket'] };
        row.querySelectorAll('input[data-k]').forEach((i) => { payload[i.dataset.k] = Number(i.value); });
        button.disabled = true;
        try {
          await api('/v1/admin/profiles/' + encodeURIComponent(payload.name), {
            method: 'PUT', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          button.textContent = 'Saved ✓';
        } catch (error) {
          button.textContent = 'Failed';
          alert(error.message);
        }
        setTimeout(() => { button.textContent = 'Save'; button.disabled = false; }, 1600);
      });
    });
  } catch (_) {}
}

async function loadEnrolments() {
  try {
    const data = await api('/v1/enrol');
    $('enrol-list').innerHTML = data.identities.length
      ? data.identities.map((i) => `
          <div class="layer-row">
            <div><div class="layer-name">${esc(i.identity)}</div>
              <div class="layer-reason">${i.samples} sample(s) · within-speaker spread ${i.spread}</div></div>
            <button class="btn ghost small" data-del="${esc(i.identity)}">Remove</button>
          </div>`).join('')
      : `<div class="empty" style="padding:16px">No enrolled identities — layer 3 will abstain,
         and the console will say so rather than reporting "genuine".</div>`;
    $('enrol-list').querySelectorAll('[data-del]').forEach((button) => {
      button.addEventListener('click', async () => {
        await api('/v1/enrol/' + encodeURIComponent(button.dataset.del), { method: 'DELETE' });
        loadEnrolments();
      });
    });
  } catch (_) {}
}

$('enrol-generate').addEventListener('click', async () => {
  const identity = $('enrol-identity').value.trim() || 'demo_cfo';
  const button = $('enrol-generate');
  button.disabled = true;
  button.textContent = 'Enrolling…';
  try {
    for (let i = 0; i < 3; i++) {
      const response = await fetch(`/v1/demo/sample?kind=bonafide&seconds=4&speaker=7&seed=${100 + i}`);
      const buffer = await response.arrayBuffer();
      // Convert to base64 in chunks: a spread over a large array blows the call stack.
      const bytes = new Uint8Array(buffer);
      let binary = '';
      for (let j = 0; j < bytes.length; j += 8192) {
        binary += String.fromCharCode.apply(null, bytes.subarray(j, j + 8192));
      }
      await api('/v1/enrol', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ identity, audio_base64: btoa(binary), encoding: 'auto' }),
      });
    }
    loadEnrolments();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
    button.textContent = 'Enrol a demo voice ×3';
  }
});

async function loadSystem() {
  try {
    const health = await api('/v1/health/full');
    $('system-stats').innerHTML = [
      ['Status', health.status],
      ['Model', health.model_loaded ? 'trained' : 'heuristic'],
      ['Sessions', health.sessions.active_sessions],
      ['Retention', health.retention.mode.replace('_', ' ')],
      ['Uptime', Math.round(health.uptime_seconds) + 's'],
    ].map(([k, v]) => `<div class="stat"><div class="k">${esc(k)}</div><div class="v" style="font-size:15px">${esc(v)}</div></div>`).join('');
  } catch (_) {}
}

$('btn-reload').addEventListener('click', async () => {
  const result = await api('/v1/admin/reload', { method: 'POST' });
  alert(result.model_loaded ? 'Model reloaded.' : 'No model artifact found — still on the heuristic detector.');
  loadHealth(); loadSystem();
});
$('btn-sweep').addEventListener('click', async () => {
  const result = await api('/v1/admin/retention/sweep', { method: 'POST' });
  alert('Retention sweep removed: ' + JSON.stringify(result.removed));
});

/* ──────────────────────────────────────────────────────────── health ── */

async function loadHealth() {
  try {
    const health = await api('/v1/health');
    const pill = $('health-pill');
    pill.className = 'pill ' + (health.status === 'ok' ? 'ok' : 'warn');
    $('health-text').textContent = health.status === 'ok' ? 'service healthy' : 'degraded';

    const modelPill = $('model-pill');
    modelPill.className = 'pill ' + (health.model_loaded ? 'ok' : 'warn');
    $('model-text').textContent = health.model_loaded ? 'trained model' : 'heuristic detector';
    modelPill.title = health.degraded.join(' · ') || 'All detectors nominal';

    $('privacy-text').textContent = health.retention.banner;
  } catch (error) {
    $('health-pill').className = 'pill bad';
    $('health-text').textContent = 'API unreachable';
  }
}

loadHealth();
loadScenarios();
loadProfiles();
refreshSessions();
setInterval(loadHealth, 15000);
