/**
 * VaniRakshak — Factual Claim Verification Console Logic
 */

// API Configuration
const API_BASE_URL = window.location.origin.includes('http') 
  ? window.location.origin 
  : 'http://localhost:8000';

// Preset Scenarios
const PRESETS = {
  weather: `Caller: Good morning sir.
Caller: It is raining heavily in Delhi right now.
Receiver: Okay, thank you for the weather update.`,

  finance: `Caller: Hello, I am calling from financial advisory desk.
Caller: RBI increased the repo rate yesterday.
Caller: You need to reallocate your portfolio immediately.`,

  scam: `Caller: Urgent alert from bank security division.
Caller: Your bank account has been blocked due to suspicious activity.
Caller: Your KYC verification has expired yesterday.
Caller: Please transfer funds to the verification escrow account immediately.`,

  call: `Caller: Hello sir, good morning.
Receiver: Yes, who is speaking?
Caller: I am calling from State Bank of India headquarters.
Caller: It is raining heavily in Delhi right now so all physical branches are closed.
Caller: Your KYC verification has expired yesterday.
Caller: RBI announced that unverified accounts will be suspended today.
Receiver: Okay, thank you for informing me.`
};

// State
let currentMode = 'text'; // 'text' | 'json'
let lastVerificationResult = null;

// DOM Elements
const apiStatusText = document.getElementById('apiStatusText');
const textModeBtn = document.getElementById('textModeBtn');
const jsonModeBtn = document.getElementById('jsonModeBtn');
const transcriptInput = document.getElementById('transcriptInput');
const inputModeLabel = document.getElementById('inputModeLabel');
const clearInputBtn = document.getElementById('clearInputBtn');
const verifyBtn = document.getElementById('verifyBtn');
const btnSpinner = document.getElementById('btnSpinner');
const btnText = document.getElementById('btnText');

// Tabs & Views
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');

// Metrics
const statSegments = document.getElementById('statSegments');
const statClaims = document.getElementById('statClaims');
const statFiltered = document.getElementById('statFiltered');
const statRiskScore = document.getElementById('statRiskScore');
const riskBadge = document.getElementById('riskBadge');

// Lists & Viewers
const claimsList = document.getElementById('claimsList');
const timelineContainer = document.getElementById('timelineContainer');
const jsonViewer = document.getElementById('jsonViewer');
const copyJsonBtn = document.getElementById('copyJsonBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  setupEventHandlers();
  // Set default scenario
  setPreset('weather');
});

// Health check
async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/health`);
    if (res.ok) {
      apiStatusText.textContent = 'API Connected (Online)';
      apiStatusText.style.color = '#34d399';
    } else {
      apiStatusText.textContent = 'API Degraded';
      apiStatusText.style.color = '#fbbf24';
    }
  } catch (err) {
    apiStatusText.textContent = 'Backend Offline';
    apiStatusText.style.color = '#f87171';
  }
}

// Event Listeners
function setupEventHandlers() {
  // Preset buttons
  document.getElementById('presetWeather').addEventListener('click', () => setPreset('weather'));
  document.getElementById('presetFinance').addEventListener('click', () => setPreset('finance'));
  document.getElementById('presetScam').addEventListener('click', () => setPreset('scam'));
  document.getElementById('presetCall').addEventListener('click', () => setPreset('call'));

  // Mode toggles
  textModeBtn.addEventListener('click', () => switchMode('text'));
  jsonModeBtn.addEventListener('click', () => switchMode('json'));

  // Clear input
  clearInputBtn.addEventListener('click', () => {
    transcriptInput.value = '';
    transcriptInput.focus();
  });

  // Verify button
  verifyBtn.addEventListener('click', runVerification);

  // Keyboard shortcut: Ctrl + Enter
  transcriptInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      runVerification();
    }
  });

  // Tab switching
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = document.getElementById(btn.dataset.tab);
      if (targetTab) targetTab.classList.add('active');
    });
  });

  // Copy JSON
  copyJsonBtn.addEventListener('click', () => {
    if (!lastVerificationResult) return;
    navigator.clipboard.writeText(JSON.stringify(lastVerificationResult, null, 2))
      .then(() => {
        copyJsonBtn.textContent = 'Copied!';
        setTimeout(() => { copyJsonBtn.textContent = 'Copy JSON'; }, 2000);
      });
  });
}

function setPreset(key) {
  if (currentMode === 'text') {
    transcriptInput.value = PRESETS[key] || '';
  } else {
    // Generate JSON structure for the preset
    const lines = (PRESETS[key] || '').split('\n').filter(l => l.trim());
    const segments = lines.map((line, idx) => {
      let speaker = null;
      let text = line;
      const match = line.match(/^([A-Za-z0-9_\s]{2,15}):\s*(.+)$/);
      if (match) {
        speaker = match[1].trim();
        text = match[2].trim();
      }
      const start = idx * 3.0;
      return {
        start: start,
        end: start + 2.5,
        text: text,
        speaker: speaker
      };
    });
    transcriptInput.value = JSON.stringify(segments, null, 2);
  }
}

function switchMode(mode) {
  if (currentMode === mode) return;
  currentMode = mode;

  if (mode === 'text') {
    textModeBtn.classList.add('active');
    jsonModeBtn.classList.remove('active');
    inputModeLabel.textContent = 'Enter transcript text or dialogue:';
    transcriptInput.placeholder = 'Type conversation text here...';
    setPreset('weather');
  } else {
    jsonModeBtn.classList.add('active');
    textModeBtn.classList.remove('active');
    inputModeLabel.textContent = 'Enter timestamped segments JSON:';
    transcriptInput.placeholder = '[\n  {\n    "start": 0.0,\n    "end": 3.0,\n    "text": "...",\n    "speaker": "Caller"\n  }\n]';
    setPreset('weather');
  }
}

// Run verification pipeline
async function runVerification() {
  const rawInput = transcriptInput.value.trim();
  if (!rawInput) {
    alert('Please enter some transcript text or select a preset scenario.');
    return;
  }

  // UI state: Loading
  setLoading(true);

  try {
    let response;

    if (currentMode === 'text') {
      response = await fetch(`${API_BASE_URL}/api/v1/claims/verify-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: rawInput })
      });
    } else {
      let parsedSegments;
      try {
        parsedSegments = JSON.parse(rawInput);
        if (!Array.isArray(parsedSegments)) throw new Error('Expected JSON array of segments');
      } catch (err) {
        throw new Error(`Invalid JSON format: ${err.message}`);
      }

      response = await fetch(`${API_BASE_URL}/api/v1/claims/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ segments: parsedSegments })
      });
    }

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: HTTP ${response.status}`);
    }

    const data = await response.json();
    lastVerificationResult = data;
    renderResults(data);
  } catch (err) {
    console.error('Verification error:', err);
    alert(`Verification failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  if (isLoading) {
    verifyBtn.disabled = true;
    btnSpinner.classList.add('active');
    btnText.textContent = 'Verifying...';
    emptyState.classList.add('hidden');
    loadingState.classList.remove('hidden');
    claimsList.innerHTML = '';
    timelineContainer.innerHTML = '';
  } else {
    verifyBtn.disabled = false;
    btnSpinner.classList.remove('active');
    btnText.textContent = 'Verify Claims';
    loadingState.classList.add('hidden');
  }
}

// Render Results
function renderResults(data) {
  emptyState.classList.add('hidden');

  const { transcript, claims, risk } = data;

  // 1. Update Metrics Ribbon
  statSegments.textContent = transcript.length;
  statClaims.textContent = claims.length;
  statFiltered.textContent = Math.max(0, transcript.length - claims.length);
  
  const riskVal = risk.claim_risk_score.toFixed(2);
  statRiskScore.textContent = riskVal;

  riskBadge.textContent = risk.risk_level;
  riskBadge.className = `risk-badge risk-${risk.risk_level.toLowerCase()}`;

  // 2. Render Tab 1: Claims & Evidence
  claimsList.innerHTML = '';
  if (claims.length === 0) {
    claimsList.innerHTML = `
      <div class="empty-state">
        <p>No verifiable factual claims detected in this transcript. Sentences were classified as conversational noise or non-factual statements.</p>
      </div>
    `;
  } else {
    claims.forEach((c) => {
      claimsList.appendChild(createClaimCard(c));
    });
  }

  // 3. Render Tab 2: Timeline
  timelineContainer.innerHTML = '';
  const claimIds = new Set(claims.map(c => `${c.start}-${c.end}`));

  transcript.forEach(seg => {
    const isClaim = claimIds.has(`${seg.start}-${seg.end}`);
    const matchedClaim = claims.find(c => Math.abs(c.start - seg.start) < 0.1 && Math.abs(c.end - seg.end) < 0.1);

    const segEl = document.createElement('div');
    segEl.className = `timeline-segment ${isClaim ? 'is-claim' : 'is-filtered'}`;

    let statusTagHtml = '';
    if (matchedClaim) {
      const v = matchedClaim.verification.status.toLowerCase();
      statusTagHtml = `<span class="segment-status-tag verdict-${v}">${matchedClaim.verification.status}</span>`;
    } else {
      statusTagHtml = `<span class="segment-status-tag" style="background: rgba(100, 116, 139, 0.2); color: #94a3b8;">FILTERED NOISE</span>`;
    }

    segEl.innerHTML = `
      <div class="segment-time">[${formatTime(seg.start)} - ${formatTime(seg.end)}]</div>
      <div class="segment-body">
        ${seg.speaker ? `<div class="segment-speaker">${escapeHtml(seg.speaker)}</div>` : ''}
        <div class="segment-text">
          "${escapeHtml(seg.text)}"
          ${statusTagHtml}
        </div>
      </div>
    `;
    timelineContainer.appendChild(segEl);
  });

  // 4. Render Tab 3: JSON Viewer
  jsonViewer.textContent = JSON.stringify(data, null, 2);
}

// Create Claim Card Component
function createClaimCard(claim) {
  const card = document.createElement('div');
  card.className = 'claim-card';

  const status = claim.verification.status;
  const statusLower = status.toLowerCase();
  const confidencePercent = Math.round(claim.verification.verification_confidence * 100);

  // Entities tags
  const sc = claim.structured_claim || {};
  const entityTags = [];
  if (sc.location) entityTags.push({ key: 'Location', val: sc.location });
  if (sc.organization) entityTags.push({ key: 'Org', val: sc.organization });
  if (sc.entity && sc.entity !== sc.organization) entityTags.push({ key: 'Entity', val: sc.entity });
  if (sc.condition) entityTags.push({ key: 'Condition', val: sc.condition });
  if (sc.time_reference) entityTags.push({ key: 'Time', val: sc.time_reference });
  if (sc.event) entityTags.push({ key: 'Event', val: sc.event });

  const entityTagsHtml = entityTags.map(e => `
    <div class="entity-tag">
      <span class="entity-key">${e.key}:</span>
      <span>${escapeHtml(e.val)}</span>
    </div>
  `).join('');

  // Evidence list
  const evidenceItems = claim.verification.evidence || [];
  const evidenceHtml = evidenceItems.map(ev => `
    <div class="evidence-item">
      <div class="evidence-header">
        <a href="${escapeHtml(ev.url)}" target="_blank" rel="noopener noreferrer" class="evidence-title">
          ${escapeHtml(ev.title || ev.url)}
        </a>
        <span class="source-badge">${escapeHtml(ev.source_type || 'WEB')}</span>
      </div>
      <div class="evidence-snippet">${escapeHtml(ev.content || '')}</div>
    </div>
  `).join('');

  card.innerHTML = `
    <div class="claim-header">
      <div class="claim-meta">
        <span class="claim-id">${claim.claim_id}</span>
        <span class="timestamp-pill">${formatTime(claim.start)} - ${formatTime(claim.end)}</span>
        <span class="type-pill">${claim.claim_type}</span>
      </div>
      <div class="verdict-badge verdict-${statusLower}">
        ${status === 'SUPPORTED' ? '&#10003; SUPPORTED' : status === 'CONTRADICTED' ? '&#10007; CONTRADICTED' : '&#63; UNVERIFIABLE'}
      </div>
    </div>

    <div class="claim-body">
      <div class="claim-text">"${escapeHtml(claim.text)}"</div>

      ${entityTagsHtml ? `<div class="entity-tags">${entityTagsHtml}</div>` : ''}

      ${claim.verification.query ? `
        <div class="query-box">
          <span class="query-label">Tavily Query:</span>
          <span>"${escapeHtml(claim.verification.query)}"</span>
        </div>
      ` : ''}

      <div class="reason-box">
        <strong>Forensic Reasoning:</strong> ${escapeHtml(claim.verification.reason)}
      </div>
    </div>

    <div class="claim-footer">
      <div>Confidence: <strong>${confidencePercent}%</strong> &bull; Risk Contribution: <strong>${claim.risk_contribution.toFixed(2)}</strong></div>
      ${evidenceItems.length > 0 ? `
        <button class="evidence-toggle-btn">
          <span>Sources (${evidenceItems.length})</span>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </button>
      ` : '<span style="font-size: 11px; color: var(--text-muted);">No external evidence found</span>'}
    </div>

    ${evidenceItems.length > 0 ? `
      <div class="evidence-list hidden">
        ${evidenceHtml}
      </div>
    ` : ''}
  `;

  // Toggle evidence list
  const toggleBtn = card.querySelector('.evidence-toggle-btn');
  const evList = card.querySelector('.evidence-list');
  if (toggleBtn && evList) {
    toggleBtn.addEventListener('click', () => {
      evList.classList.toggle('hidden');
      toggleBtn.querySelector('svg').style.transform = evList.classList.contains('hidden') ? 'rotate(0deg)' : 'rotate(180deg)';
    });
  }

  return card;
}

// Helpers
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
