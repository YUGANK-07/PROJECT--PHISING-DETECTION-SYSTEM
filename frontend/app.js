/* ─── PhishGuard frontend logic ─── */
'use strict';

const API = 'http://localhost:8000';
const KEY = 'Bearer demo-key-phishguard-2024';
const H   = { 'Content-Type': 'application/json', 'Authorization': KEY };

const $  = id => document.getElementById(id);

// ── Health check ──────────────────────────────────────────────────────────────
async function ping() {
  const dot   = $('apiDot');
  const label = $('apiLabel');
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3500) });
    const d = await r.json();
    if (d.status === 'ok') {
      dot.className   = 'api-dot ok';
      label.textContent = 'API Online';
    } else {
      dot.className   = 'api-dot err';
      label.textContent = 'Degraded';
    }
  } catch {
    dot.className   = 'api-dot err';
    label.textContent = 'Offline';
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg, type = '') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const icon = type === 'err' ? '⚠' : '✓';
  el.innerHTML = `<span>${icon}</span><span>${msg}</span>`;
  $('toastStack').appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(20px)';
    el.style.transition = 'opacity .3s, transform .3s';
    setTimeout(() => el.remove(), 350);
  }, 3200);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const risk2cls   = r => r === 'High' ? 'high' : r === 'Medium' ? 'med' : 'low';
const risk2color = r => r === 'High' ? '#f43f5e' : r === 'Medium' ? '#f59e0b' : '#10b981';
const risk2emoji = r => r === 'High' ? '🚨' : r === 'Medium' ? '⚠️' : '✅';
const risk2label = r => r === 'High' ? '🚨 High Risk' : r === 'Medium' ? '⚠️ Medium Risk' : '✅ Low Risk';

// ── Input wiring ──────────────────────────────────────────────────────────────
const urlInput = $('urlInput');
const clearX   = $('clearX');

urlInput.addEventListener('input', () => {
  clearX.classList.toggle('show', urlInput.value.length > 0);
});
clearX.addEventListener('click', () => {
  urlInput.value = '';
  clearX.classList.remove('show');
  urlInput.focus();
});

const refInput = $('refInput');
const refClearX = $('refClearX');
refInput.addEventListener('input', () => {
  refClearX.classList.toggle('show', refInput.value.length > 0);
});
refClearX.addEventListener('click', () => {
  refInput.value = '';
  refClearX.classList.remove('show');
  refInput.focus();
});

let currentTargetUrl = '';

// ── Scan ──────────────────────────────────────────────────────────────────────
async function scan(url) {
  url = url.trim();
  if (!url) { toast('Please enter a URL', 'err'); return; }
  if (!/^https?:\/\//i.test(url)) url = 'https://' + url;

  showScanning();

  try {
    const bodyData = { url, include_explanation: true };
    const refUrl = refInput.value.trim();
    if (refUrl) bodyData.reference_url = refUrl;

    const res = await fetch(`${API}/predict`, {
      method: 'POST', headers: H,
      body: JSON.stringify(bodyData),
      signal: AbortSignal.timeout(30_000),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    hideScanning();
    showResult(data);
  } catch (e) {
    hideScanning();
    toast(e.message, 'err');
    showScanner();
  }
}

// ── View transitions ──────────────────────────────────────────────────────────
function showScanning() {
  $('viewScanner').style.display  = 'none';
  $('viewResult').style.display   = 'none';
  $('scanOverlay').style.display  = 'flex';
  $('scanBtn').disabled = true;
}
function hideScanning() {
  $('scanOverlay').style.display = 'none';
  $('scanBtn').disabled = false;
}
function showScanner() {
  $('viewResult').style.display  = 'none';
  $('viewScanner').style.display = 'flex';
}

// ── Render result ─────────────────────────────────────────────────────────────
function showResult(d) {
  const { url, phishing_probability: prob, risk_level: risk,
          explanation, processing_time_ms: ms, cached } = d;

  const pct   = Math.round(prob * 100);
  const cls   = risk2cls(risk);
  const color = risk2color(risk);
  const circumference = 2 * Math.PI * 66; // r=66 → ≈415

  // Verdict hero class
  $('verdictHero').className = `verdict-hero risk-${cls}`;

  // Badge
  const badge = $('verdictBadge');
  badge.className = `verdict-risk-badge ${cls}`;
  badge.textContent = risk2label(risk);

  // Emoji + percentage
  $('verdictEmoji').textContent = risk2emoji(risk);
  $('verdictPct').textContent   = `${pct}%`;
  $('verdictPct').style.color   = color;

  // URL text
  const urlEl = $('verdictUrlText');
  urlEl.textContent = url.length > 72 ? url.slice(0, 69) + '…' : url;

  // Meta
  $('vmTime').textContent  = `${Math.round(ms)}ms`;
  $('vmCache').textContent = cached ? '⚡ cached' : 'live analysis';

  // Animate score ring after paint
  const fill = $('vrFill');
  fill.style.stroke = color;
  fill.style.strokeDashoffset = circumference;  // reset
  requestAnimationFrame(() => requestAnimationFrame(() => {
    fill.style.strokeDashoffset = circumference - (prob * circumference);
  }));

  // Signals
  const sigSec  = $('signalsSection');
  const sigList = $('signalsList');
  sigList.innerHTML = '';

  if (explanation && explanation.length) {
    sigSec.style.display = 'block';
    explanation.slice(0, 7).forEach((f, i) => {
      const bad  = f.impact === 'increases_risk';
      const shapVal = f.shap_value != null ? (f.shap_value > 0 ? `+${f.shap_value.toFixed(3)}` : f.shap_value.toFixed(3)) : '';

      const row = document.createElement('div');
      row.className = 'signal-row';
      row.style.animationDelay = `${i * 60}ms`;
      row.innerHTML = `
        <div class="signal-indicator ${bad ? 'bad' : 'good'}"></div>
        <div class="signal-body">
          <div class="signal-label">${f.human_label}</div>
          <div class="signal-feature">${f.feature} = ${f.value}</div>
        </div>
        ${shapVal ? `<div class="signal-shap ${bad ? 'pos' : 'neg'}">${shapVal}</div>` : ''}
      `;
      sigList.appendChild(row);
    });
  } else {
    sigSec.style.display = 'none';
  }

  currentTargetUrl = url;
  $('viewScanner').style.display = 'none';
  $('viewResult').style.display  = 'flex';
}

// ── Events ────────────────────────────────────────────────────────────────────
$('scanBtn').addEventListener('click', () => scan(urlInput.value));
urlInput.addEventListener('keydown', e => { if (e.key === 'Enter') scan(urlInput.value); });

$('backBtn').addEventListener('click', () => {
  showScanner();
  urlInput.value = '';
  clearX.classList.remove('show');
  setTimeout(() => urlInput.focus(), 100);
});

// ── Secure Preview Logic ──────────────────────────────────────────────────────
const previewBtn = $('previewBtn');
const previewModal = $('previewModal');
const closePreviewBtn = $('closePreviewBtn');
const previewLoader = $('previewLoader');
const previewImageWrap = $('previewImageWrap');
const previewImage = $('previewImage');

previewBtn.addEventListener('click', async () => {
  if (!currentTargetUrl) return;
  
  // Reset and show modal
  previewModal.style.display = 'flex';
  previewLoader.style.display = 'flex';
  previewImageWrap.style.display = 'none';
  previewImage.src = '';
  
  try {
    const res = await fetch(`${API}/preview`, {
      method: 'POST',
      headers: H,
      body: JSON.stringify({ url: currentTargetUrl }),
      signal: AbortSignal.timeout(15_000), // 15s timeout for playwright
    });
    
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    
    const data = await res.json();
    
    // Hide loader, show image
    previewLoader.style.display = 'none';
    previewImageWrap.style.display = 'flex';
    previewImage.src = `data:image/png;base64,${data.image_b64}`;
    toast(`Preview rendered in ${Math.round(data.processing_time_ms)}ms`, 'ok');
  } catch (e) {
    previewModal.style.display = 'none';
    toast('Preview Failed: ' + e.message, 'err');
  }
});

closePreviewBtn.addEventListener('click', () => {
  previewModal.style.display = 'none';
});

// Close modal when clicking outside content
previewModal.addEventListener('click', (e) => {
  if (e.target === previewModal) {
    previewModal.style.display = 'none';
  }
});

document.querySelectorAll('.chip').forEach(c =>
  c.addEventListener('click', () => scan(c.dataset.url))
);

// ── Typewriter placeholder ────────────────────────────────────────────────────
const PLACEHOLDERS = [
  'https://paypal-secure-verify.account.xyz/login.php',
  'https://rnicrosoft.com',
  'https://paypal.com/signin',
  'https://g00gle-auth.net/token?id=abc123',
  'https://apple.com/account',
];
let phIdx = 0, phChar = 0, phDir = 1, phTimer;
function cyclePlaceholder() {
  const target = PLACEHOLDERS[phIdx];
  if (phDir === 1) {
    phChar++;
    if (phChar > target.length) { phDir = -1; phTimer = setTimeout(cyclePlaceholder, 1800); return; }
  } else {
    phChar--;
    if (phChar < 0) { phChar = 0; phDir = 1; phIdx = (phIdx + 1) % PLACEHOLDERS.length; }
  }
  urlInput.placeholder = target.slice(0, phChar) + (phDir === 1 && phChar < target.length ? '|' : '');
  phTimer = setTimeout(cyclePlaceholder, phDir === 1 ? 48 : 22);
}
cyclePlaceholder();
urlInput.addEventListener('focus',  () => clearTimeout(phTimer));
urlInput.addEventListener('blur',   () => { phChar = 0; phDir = 1; cyclePlaceholder(); });

// ── Init ──────────────────────────────────────────────────────────────────────
ping();
setInterval(ping, 30_000);
