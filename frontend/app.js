// ═══════════════════════════════════════════════════════════════
// app.js — Prompt Builder IDE • Frontend Controller (Phase 3)
// ═══════════════════════════════════════════════════════════════
const API = '';  // Same-origin when served by FastAPI

// ── App State ──
const appState = {
    lastRunResult: null,
    compareSlotA: null,
    compareSlotB: null,
    abortController: null,
    ollamaReachable: false,
};

// ── DOM Refs ──
const $ = id => document.getElementById(id);
const statusDot = $('status-dot');
const statusText = $('status-text');
const btnRun = $('btn-run');
const btnCompile = $('btn-compile');
const btnCompare = $('btn-compare');

// ═════════════════════════════════════════
//  TAB SWITCHING
// ═════════════════════════════════════════
const panes = ['inputs', 'config', 'constraints', 'loop'];
const sidebarItems = document.querySelectorAll('.sidebar-item[onclick*="switchTab"]');
const topTabs = document.querySelectorAll('.tab');

function switchTab(name, sidebarEl, tabEl) {
    panes.forEach(p => {
        const el = document.getElementById('pane-' + p);
        if (el) { el.style.display = p === name ? 'flex' : 'none'; }
    });
    const target = document.getElementById('pane-' + name);
    if (target) { target.style.display = 'flex'; target.style.flexDirection = 'column'; target.style.gap = '10px'; }
    sidebarItems.forEach(el => el.classList.remove('active'));
    if (sidebarEl) sidebarEl.classList.add('active');
    topTabs.forEach(t => t.classList.remove('active'));
    const matchTab = document.getElementById('tab-' + name);
    if (matchTab) matchTab.classList.add('active');
}

// Init default pane
const paneInputs = document.getElementById('pane-inputs');
if (paneInputs) { paneInputs.style.display = 'flex'; paneInputs.style.flexDirection = 'column'; paneInputs.style.gap = '10px'; }

// ── Output tab switching ──
const outContents = document.querySelectorAll('.out-content');
const outTabEls = document.querySelectorAll('.out-tab');

function switchOutTab(name, tabEl) {
    outContents.forEach(el => el.classList.remove('active'));
    outTabEls.forEach(el => el.classList.remove('active'));
    const outEl = document.getElementById('out-' + name);
    if (outEl) outEl.classList.add('active');
    if (tabEl) tabEl.classList.add('active');
    else {
        outTabEls.forEach(el => {
            if (el.textContent.toLowerCase().trim().startsWith(name.substring(0, 3))) el.classList.add('active');
        });
    }
}

function togglePanel(head) { head.parentElement.classList.toggle('collapsed'); }

// ═════════════════════════════════════════
//  TAG AREA HELPERS
// ═════════════════════════════════════════
function getTagValues(areaId) {
    const area = $(areaId);
    if (!area) return [];
    const tags = area.querySelectorAll('.tag');
    const values = [];
    tags.forEach(t => {
        const text = t.childNodes[0]?.textContent?.trim();
        if (text) values.push(text);
    });
    return values;
}

function getVariables() {
    const container = $('var-rows');
    if (!container) return {};
    const rows = container.querySelectorAll('.var-row');
    const vars = {};
    rows.forEach(row => {
        const inputs = row.querySelectorAll('.f-input');
        const key = inputs[0]?.value?.trim();
        const val = inputs[1]?.value?.trim();
        if (key && val) vars[key] = val;
    });
    return vars;
}

function getExamples() {
    const container = $('shot-pairs');
    if (!container) return [];
    const pairs = container.querySelectorAll('.shot-pair');
    const examples = [];
    pairs.forEach(pair => {
        const textareas = pair.querySelectorAll('.f-textarea');
        const inp = textareas[0]?.value?.trim();
        const out = textareas[1]?.value?.trim();
        if (inp || out) examples.push({ input: inp || '', output: out || '' });
    });
    return examples;
}

// ═════════════════════════════════════════
//  BUILD PAYLOAD
// ═════════════════════════════════════════
function buildPayload() {
    const userInput = $('input-rough-concept')?.value?.trim();
    if (!userInput) { alert('Please enter a rough concept.'); return null; }

    const payload = { user_input: userInput };
    payload.framework = $('select-framework')?.value || 'co_star';
    payload.domain = $('select-domain')?.value || 'General';
    payload.target_model = $('select-target-model')?.value || 'llama3:8b';
    payload.expected_format = $('select-response-format')?.value || 'plain_text';

    const maxTokens = parseInt($('input-max-tokens')?.value);
    if (!isNaN(maxTokens)) payload.max_tokens = maxTokens;

    const temp = parseInt($('input-temperature')?.value);
    if (!isNaN(temp)) payload.temperature = temp / 100;

    const maxIter = parseInt($('input-max-iterations')?.value);
    if (!isNaN(maxIter)) payload.max_iterations = maxIter;

    const threshold = parseFloat($('input-quality-threshold')?.value);
    if (!isNaN(threshold)) payload.quality_threshold = Math.round(threshold);

    // Constraints
    const mustInclude = getTagValues('tag-area-must-include');
    if (mustInclude.length) payload.must_include = mustInclude;

    const mustNot = getTagValues('tag-area-must-not');
    if (mustNot.length) payload.must_not_include = mustNot;

    const constraints = getTagValues('tag-area-constraints');
    if (constraints.length) payload.constraints = constraints;

    const variables = getVariables();
    if (Object.keys(variables).length) payload.variables = variables;

    const examples = getExamples();
    if (examples.length) payload.examples = examples;

    return payload;
}

// ═════════════════════════════════════════
//  STATUS INDICATOR
// ═════════════════════════════════════════
function setStatus(state, text) {
    statusDot.className = 'status-dot ' + state;
    statusText.textContent = text;
}

async function checkHealth() {
    try {
        const resp = await fetch(API + '/api/health');
        const data = await resp.json();
        appState.ollamaReachable = data.ollama_reachable;
        if (data.ollama_reachable) {
            setStatus('green', 'Ollama connected');
        } else {
            setStatus('red', 'Ollama unreachable: ' + (data.detail || 'unknown'));
        }
    } catch (e) {
        setStatus('red', 'API unreachable');
        appState.ollamaReachable = false;
    }
}

// ═════════════════════════════════════════
//  LOADING STATE
// ═════════════════════════════════════════
function setLoading(btn, loading) {
    if (loading) {
        btn.dataset.origText = btn.textContent;
        btn.classList.add('btn-disabled');
        btn.innerHTML = '<span class="spinner active"></span> WORKING…';
    } else {
        btn.classList.remove('btn-disabled');
        btn.textContent = btn.dataset.origText || btn.textContent;
    }
}

// ═════════════════════════════════════════
//  RENDER OUTPUT PANELS
// ═════════════════════════════════════════
function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function renderPromptOutput(compiledText, schema) {
    const el = $('out-prompt');
    if (!el) return;
    const fw = schema?.framework_name || '';
    const model = schema?.target_model || '';
    el.innerHTML = `
    <div class="data-block-title" style="font-size:8px;letter-spacing:2px;color:var(--text-mute);text-transform:uppercase;margin-bottom:10px;">Compiled Prompt</div>
    <div class="prompt-code"><span class="cmt"># ${escapeHTML(fw)} · ${escapeHTML(model)}</span>\n\n${escapeHTML(compiledText)}</div>`;
}

function renderScoreOutput(score) {
    const el = $('out-score');
    if (!el || !score) return;
    const overall = score.overall_score || 0;
    const pct = overall * 10;
    el.innerHTML = `
    <div class="score-ring-wrap">
      <div class="score-ring" style="background:conic-gradient(var(--cyan) 0% ${pct}%, var(--border2) ${pct}% 100%);">
        <span class="score-ring-num">${overall}</span>
      </div>
      <div class="score-ring-label">Quality Score</div>
    </div>
    <div class="score-bars">
      ${renderBar('Clarity', score.clarity_score)}
      ${renderBar('Specificity', score.specificity_score)}
      ${renderBar('Structure', score.structure_score)}
      ${renderBar('Constraints', score.constraint_score)}
      ${renderBar('Token Efficiency', score.token_efficiency_score)}
    </div>`;
}

function renderBar(name, val) {
    const v = val || 0;
    return `<div class="s-bar-row"><div class="s-bar-top"><span class="s-bar-name">${name}</span><span class="s-bar-num">${v}</span></div><div class="s-bar-track"><div class="s-bar-fill" style="width:${v * 10}%"></div></div></div>`;
}

function renderFeedback(score) {
    const el = $('out-feedback');
    if (!el || !score) return;
    const strengths = (score.strengths || []).map(s => `<div><span class="ok">✓</span> ${escapeHTML(s)}</div>`).join('');
    const issues = (score.issues || []).map(s => `<div><span class="warn">⚠</span> ${escapeHTML(s)}</div>`).join('');
    const suggestions = (score.suggestions || []).map(s => `<div><span class="key">→</span> ${escapeHTML(s)}</div>`).join('');
    el.innerHTML = `
    <div class="data-block"><div class="data-block-title">Strengths</div>${strengths || '<div style="color:var(--text-mute)">—</div>'}</div>
    <div class="data-block"><div class="data-block-title">Issues</div>${issues || '<div style="color:var(--text-mute)">—</div>'}</div>
    <div class="data-block"><div class="data-block-title">Suggestions</div>${suggestions || '<div style="color:var(--text-mute)">—</div>'}</div>`;
}

function renderSimulation(testResult) {
    const el = $('out-sim');
    if (!el) return;
    if (!testResult) { el.innerHTML = '<div class="data-block"><div class="data-block-title">No simulation data</div></div>'; return; }
    el.innerHTML = `
    <div class="data-block">
      <div class="data-block-title">Simulation Result</div>
      <div><span class="key">format_comply: </span><span class="${testResult.follows_format ? 'ok' : 'err'}">${testResult.follows_format}</span></div>
      <div><span class="key">tokens: </span><span class="val">${testResult.token_count || '—'}</span></div>
      <div><span class="key">exec_time: </span><span class="val">${testResult.execution_time_ms ? (testResult.execution_time_ms / 1000).toFixed(2) + 's' : '—'}</span></div>
      <div><span class="key">model: </span><span class="val">${escapeHTML(testResult.model_used || '—')}</span></div>
    </div>
    <div class="data-block">
      <div class="data-block-title">Model Response Preview</div>
      <div style="color:var(--text-mute);line-height:1.7;white-space:pre-wrap;">${escapeHTML(testResult.model_response || '—')}</div>
    </div>`;
}

function renderHistory(iterations) {
    const el = $('out-history');
    if (!el) return;
    if (!iterations || !iterations.length) { el.innerHTML = '<div style="color:var(--text-mute);font-size:10px;">No iterations yet.</div>'; return; }
    let html = '<div class="iter-timeline" style="margin-left:5px;">';
    iterations.forEach((it, i) => {
        const isCurrent = i === iterations.length - 1;
        html += `<div class="iter-node${isCurrent ? ' current' : ''}">
      <div class="iter-head${isCurrent ? '' : ' done'}">ITER ${it.iteration_number} · score ${it.overall_score}</div>
      <div class="iter-body">${escapeHTML(it.action_taken || it.critique || '—')}</div>
    </div>`;
    });
    html += '</div>';
    el.innerHTML = html;
}

function renderAB(slotA, slotB) {
    const el = $('out-ab');
    if (!el) return;
    if (!slotA && !slotB) { el.innerHTML = '<div style="color:var(--text-mute);font-size:10px;">Run twice to fill A/B slots.</div>'; return; }
    const aScore = slotA?.final_score?.overall_score || 0;
    const bScore = slotB?.final_score?.overall_score || 0;
    const aWin = aScore >= bScore;
    el.innerHTML = `
    <div style="font-size:8px;letter-spacing:2px;color:var(--text-mute);text-transform:uppercase;margin-bottom:10px;">Prompt Comparison</div>
    <div class="ab-grid">
      <div class="ab-col${aWin ? ' winner' : ''}">
        <div class="ab-head"><span>Prompt A</span><span class="${aWin ? 'score' : ''}">${aScore}${aWin ? ' ✓' : ''}</span></div>
        <div class="ab-body" style="white-space:pre-wrap;max-height:200px;overflow-y:auto;">${escapeHTML(slotA?.final_prompt || '—')}</div>
      </div>
      <div class="ab-col${!aWin ? ' winner' : ''}">
        <div class="ab-head"><span>Prompt B</span><span class="${!aWin ? 'score' : ''}">${bScore}${!aWin ? ' ✓' : ''}</span></div>
        <div class="ab-body" style="white-space:pre-wrap;max-height:200px;overflow-y:auto;">${escapeHTML(slotB?.final_prompt || '—')}</div>
      </div>
    </div>`;
}

// ═════════════════════════════════════════
//  SIDEBAR META UPDATE
// ═════════════════════════════════════════
function updateSidebarMeta(data) {
    const meta = document.querySelector('.sidebar-meta');
    if (!meta || !data) return;
    const score = data.final_score?.overall_score || '—';
    const iters = data.iterations_history?.length || 0;
    const maxIter = $('input-max-iterations')?.value || 3;
    const model = $('select-target-model')?.value || '—';
    meta.innerHTML = `iter ${iters} / ${maxIter} &nbsp;·&nbsp; score ${score}<br>model ${escapeHTML(model)}<br>status: ${escapeHTML(data.status || '—')}`;
}

// ═════════════════════════════════════════
//  TITLEBAR PILL
// ═════════════════════════════════════════
function setPill(state) {
    const pill = document.querySelector('.titlebar-pill');
    if (!pill) return;
    if (state === 'refining') {
        pill.className = 'titlebar-pill pill-refining';
        pill.textContent = '● REFINING';
    } else {
        pill.className = 'titlebar-pill pill-idle';
        pill.textContent = '○ IDLE';
    }
}

// ═════════════════════════════════════════
//  API CALLS
// ═════════════════════════════════════════
async function apiCall(endpoint, payload, btn) {
    if (appState.abortController) appState.abortController.abort();
    appState.abortController = new AbortController();

    setLoading(btn, true);
    setPill('refining');
    setStatus('green', 'Processing…');

    try {
        const resp = await fetch(API + endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: appState.abortController.signal,
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }
        return await resp.json();
    } finally {
        setLoading(btn, false);
        setPill('idle');
        appState.abortController = null;
    }
}

// ── COMPILE ──
async function handleCompile() {
    const payload = buildPayload();
    if (!payload) return;
    try {
        const data = await apiCall('/api/compile', payload, btnCompile);
        renderPromptOutput(data.compiled, data.prompt_schema);
        switchOutTab('prompt');
        setStatus('green', 'Compiled successfully');
    } catch (e) {
        if (e.name === 'AbortError') return;
        setStatus('red', 'Compile failed: ' + e.message);
        alert('Compile error: ' + e.message);
    }
}

// ── RUN ──
async function handleRun() {
    const payload = buildPayload();
    if (!payload) return;
    try {
        const data = await apiCall('/api/run', payload, btnRun);
        appState.lastRunResult = data;

        // Rotate A/B
        appState.compareSlotA = appState.compareSlotB;
        appState.compareSlotB = data;

        // Render all panels
        renderPromptOutput(data.final_prompt, data.final_prompt_schema);
        renderScoreOutput(data.final_score);
        renderFeedback(data.final_score);
        renderSimulation(data.test_result);
        renderHistory(data.iterations_history);
        renderAB(appState.compareSlotA, appState.compareSlotB);
        updateSidebarMeta(data);

        switchOutTab('score');
        setStatus('green', 'Run complete · score ' + (data.final_score?.overall_score || '—'));
    } catch (e) {
        if (e.name === 'AbortError') return;
        setStatus('red', 'Run failed: ' + e.message);
        alert('Run error: ' + e.message);
    }
}

// ── COMPARE A/B ──
async function handleCompare() {
    const payload = buildPayload();
    if (!payload) return;
    try {
        const data = await apiCall('/api/run', payload, btnCompare);
        appState.compareSlotA = appState.compareSlotB || appState.lastRunResult;
        appState.compareSlotB = data;
        appState.lastRunResult = data;

        renderAB(appState.compareSlotA, appState.compareSlotB);
        switchOutTab('ab');
        setStatus('green', 'A/B comparison ready');
    } catch (e) {
        if (e.name === 'AbortError') return;
        setStatus('red', 'Compare failed: ' + e.message);
    }
}

// ═════════════════════════════════════════
//  WIRE BUTTONS
// ═════════════════════════════════════════
if (btnRun) btnRun.addEventListener('click', handleRun);
if (btnCompile) btnCompile.addEventListener('click', handleCompile);
if (btnCompare) btnCompare.addEventListener('click', handleCompare);

// ═════════════════════════════════════════
//  TAG ADD/REMOVE INTERACTIVITY
// ═════════════════════════════════════════
document.addEventListener('click', (e) => {
    // Remove tag
    if (e.target.classList.contains('tx')) {
        e.target.parentElement.remove();
        return;
    }
    // Add tag
    if (e.target.classList.contains('add-tag')) {
        const area = e.target.closest('.tag-area') || e.target.previousElementSibling;
        if (!area) return;
        const val = prompt('Enter value:');
        if (!val || !val.trim()) return;
        const tag = document.createElement('div');
        tag.className = 'tag';
        tag.innerHTML = `${escapeHTML(val.trim())} <span class="tx">✕</span>`;
        const addBtn = area.querySelector('.add-tag') || e.target;
        area.insertBefore(tag, addBtn);
    }
});

// ═════════════════════════════════════════
//  INIT
// ═════════════════════════════════════════
setPill('idle');
checkHealth();
setInterval(checkHealth, 30000);
