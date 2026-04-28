/* =================================================================
   FlairSim Benchmark -- Vanilla JS SPA  (Phase 1.5)
   ================================================================= */

'use strict';

// --- Config ---
const API_BASE = '/api';

// --- Global State ---
let currentView = 'landing';
let session = null;       // current Session object from API
let scenarios = [];       // cached scenario list
let gameEngine = null;    // GameEngine instance
let trajectory = [];      // array of {x, y, z, step}
let stepsDetail = [];     // array of {dx, dy, dz, action_type, reason}
let startTime = null;
let lastObservation = null;
let currentRunDetailId = null;  // tracks active run-detail view
let _startingSession = false;   // true while session creation is in progress
let _spectatorMode = false;     // true when viewing someone else's session
let _processesRefreshTimer = null; // polling timer for active processes
let _currentModeFilter = 'ai';  // 'ai', 'human', or '' (both)

// ===================================================================
// Router
// ===================================================================

function navigateTo(view, params) {
    // Hide results modal if navigating away from play
    if (view !== 'play') {
        const modal = document.getElementById('results-modal');
        modal.hidden = true;
    }

    // Clear run-detail context when leaving that view
    if (view !== 'run-detail') {
        currentRunDetailId = null;
    }

    // Stop processes polling when leaving that view
    if (view !== 'processes' && _processesRefreshTimer) {
        clearInterval(_processesRefreshTimer);
        _processesRefreshTimer = null;
    }

    document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
    const target = document.getElementById('view-' + view);
    if (target) {
        target.classList.add('active');
        // Re-trigger animation
        target.style.animation = 'none';
        target.offsetHeight; // reflow
        target.style.animation = '';
    } else {
        // Unknown view — fall back to landing
        navigateTo('landing');
        return;
    }
    currentView = view;

    if (view === 'landing') {
        loadScenarios();
    } else if (view === 'processes') {
        loadProcesses();
    } else if (view === 'results') {
        loadResultsPage();
    } else if (view === 'about') {
        loadAboutPage();
    }

    // Preserve full hash for deep-linked views (set by handleRoute)
    const currentHash = window.location.hash.replace('#', '');
    if (!currentHash.startsWith(view + '/') && currentHash !== view) {
        window.location.hash = view;
    }
}

function handleRoute() {
    const hash = window.location.hash.replace('#', '') || 'landing';
    // Protected views -- redirect if missing context
    if (hash === 'play' && !session && !_startingSession) {
        navigateTo('landing');
        return;
    }
    if (hash === 'run-detail' && !currentRunDetailId) {
        navigateTo('landing');
        return;
    }
    // Agent page: #agent/{name}
    if (hash.startsWith('agent/')) {
        const agentName = decodeURIComponent(hash.substring(6));
        navigateTo('agent');
        loadAgentPage(agentName);
        return;
    }
    // Handle results/scenario_id
    if (hash.startsWith('results/')) {
        const scenarioId = decodeURIComponent(hash.substring(8));
        navigateTo('results');
        setTimeout(() => {
            switchResultsTab('scenario');
            const select = document.getElementById('res-filter-scenario');
            if (select) {
                select.value = scenarioId;
                loadScenarioResults(scenarioId);
            }
        }, 50);
        return;
    }
    navigateTo(hash);
}

// ===================================================================
// API Client
// ===================================================================

async function apiFetch(path, options = {}) {
    const url = API_BASE + path;
    const resp = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
}

async function fetchScenarios() {
    const data = await apiFetch('/scenarios');
    return data.scenarios || [];
}

async function createSession(scenarioId, mode, playerName, modelInfo) {
    return apiFetch('/sessions', {
        method: 'POST',
        body: JSON.stringify({
            scenario_id: scenarioId,
            mode: mode,
            player_name: playerName || null,
            model_info: modelInfo || null,
        }),
    });
}

async function deleteSession(sessionId) {
    return apiFetch('/sessions/' + sessionId, { method: 'DELETE' });
}

async function simReset(sessionId) {
    return apiFetch('/sessions/' + sessionId + '/sim/reset', { method: 'POST' });
}

async function simStep(sessionId, action) {
    return apiFetch('/sessions/' + sessionId + '/sim/step', {
        method: 'POST',
        body: JSON.stringify(action),
    });
}

async function simConfig(sessionId) {
    return apiFetch('/sessions/' + sessionId + '/sim/config');
}

async function submitToLeaderboard(runData) {
    return apiFetch('/leaderboard/submit', {
        method: 'POST',
        body: JSON.stringify(runData),
    });
}

async function fetchLeaderboardRun(runId) {
    return apiFetch('/leaderboard/' + runId);
}

async function fetchSessions() {
    return apiFetch('/sessions');
}

// ===================================================================
// Scenario Picker  (Landing)
// ===================================================================

async function loadScenarios() {
    const grid = document.getElementById('scenario-grid');
    try {
        scenarios = await fetchScenarios();
        if (scenarios.length === 0) {
            grid.innerHTML = '<div class="empty-msg">No scenarios found.</div>';
            return;
        }
        grid.innerHTML = scenarios.map(s => renderScenarioCard(s)).join('');
    } catch (err) {
        grid.innerHTML = '<div class="empty-msg">Error loading scenarios: ' + escapeHtml(err.message) + '</div>';
    }
}

function renderScenarioCard(s) {
    const id = s.scenario_id || s.id;
    const name = s.name || id;
    const desc = s.description || '';
    const domain = s.dataset ? (s.dataset.domain || '') : '';
    const roi = s.dataset ? (s.dataset.roi || '') : '';
    const modalities = s.dataset && s.dataset.modalities ? s.dataset.modalities : [];
    const maxSteps = s.max_steps || '?';
    const environment = s.environment || [];
    const difficulty = s.difficulty || 0;

    // Difficulty stars
    let starsHtml = '';
    if (difficulty > 0) {
        starsHtml = '<span class="difficulty-stars">';
        for (let i = 1; i <= 3; i++) {
            if (i <= difficulty) {
                starsHtml += '<span class="star-filled">&#9733;</span>';
            } else {
                starsHtml += '<span>&#9734;</span>';
            }
        }
        starsHtml += '</span>';
    }

    // Specs table with badge values
    let specsHtml = '<table class="scenario-specs">';
    if (domain) specsHtml += '<tr><td class="spec-label">Domain</td><td class="spec-value"><span class="tag tag-domain">' + escapeHtml(domain) + '</span></td></tr>';
    if (roi) specsHtml += '<tr><td class="spec-label">ROI</td><td class="spec-value"><span class="tag tag-roi">' + escapeHtml(roi) + '</span></td></tr>';
    specsHtml += '<tr><td class="spec-label">Environment</td><td class="spec-value">';
    if (environment.length > 0) {
        environment.forEach(e => { specsHtml += '<span class="tag tag-env">' + escapeHtml(e) + '</span> '; });
    } else {
        specsHtml += '-';
    }
    specsHtml += '</td></tr>';
    specsHtml += '<tr><td class="spec-label">Modalities</td><td class="spec-value">';
    if (modalities.length > 0) {
        modalities.forEach(m => { specsHtml += '<span class="tag tag-mod">' + escapeHtml(m) + '</span> '; });
    } else {
        specsHtml += '-';
    }
    specsHtml += '</td></tr>';
    specsHtml += '<tr><td class="spec-label">Max Steps</td><td class="spec-value"><span class="tag tag-steps">' + escapeHtml(String(maxSteps)) + '</span></td></tr>';
    specsHtml += '</table>';

    const thumbnailUrl = API_BASE + '/scenarios/' + encodeURIComponent(id) + '/thumbnail?size=300';

    return `
        <div class="scenario-card">
            <h3>${escapeHtml(name)} ${starsHtml}</h3>
            <div class="scenario-thumbnail">
                <img src="${thumbnailUrl}" alt="${escapeAttr(name)}" loading="lazy" onerror="this.parentElement.classList.add('thumb-error')">
            </div>
            <p class="scenario-desc">${escapeHtml(desc)}</p>
            ${specsHtml}
            <div class="scenario-actions">
                <button class="btn btn-primary" onclick="startPlay('${escapeAttr(id)}')">
                    Play
                </button>
                <button class="btn btn-ghost" onclick="navigateTo('results');setTimeout(()=>{switchResultsTab('scenario');document.getElementById('res-filter-scenario').value='${escapeAttr(id)}';loadScenarioResults('${escapeAttr(id)}')},50)">
                    View Results
                </button>
            </div>
        </div>
    `;
}

// ===================================================================
// Play Session Management
// ===================================================================

let _pendingPlayScenarioId = null;

async function startPlay(scenarioId) {
    // Show pre-play modal to ask for player name
    _pendingPlayScenarioId = scenarioId;
    const nameInput = document.getElementById('preplay-player-name');
    nameInput.value = localStorage.getItem('flairsim-player-name') || '';
    document.getElementById('preplay-modal').hidden = false;
    nameInput.focus();
}

function cancelPreplay() {
    document.getElementById('preplay-modal').hidden = true;
    _pendingPlayScenarioId = null;
}

async function confirmStartPlay() {
    const nameInput = document.getElementById('preplay-player-name');
    const playerName = nameInput.value.trim();
    if (playerName) {
        localStorage.setItem('flairsim-player-name', playerName);
    }
    document.getElementById('preplay-modal').hidden = true;

    const scenarioId = _pendingPlayScenarioId;
    _pendingPlayScenarioId = null;
    if (!scenarioId) return;

    try {
        _startingSession = true;
        _spectatorMode = false;
        navigateTo('play');
        setPlayStatus('starting');

        const scenarioData = scenarios.find(s => (s.scenario_id || s.id) === scenarioId);
        document.getElementById('play-scenario-name').textContent =
            scenarioData ? (scenarioData.name || scenarioId) : scenarioId;
        document.getElementById('play-scenario-title').textContent =
            scenarioData ? (scenarioData.name || scenarioId) : scenarioId;
        document.getElementById('play-scenario-desc').textContent =
            scenarioData ? (scenarioData.description || '') : '';

        const badge = document.getElementById('play-mode-badge');
        badge.textContent = 'HUMAN';
        badge.className = 'badge';

        // Show controls, hide spectator badge
        document.getElementById('controls-info').style.display = '';
        document.getElementById('play-spectator-badge').hidden = true;

        // Create session with player name
        session = await createSession(scenarioId, 'human', playerName || null);
        _startingSession = false;
        setPlayStatus(session.status);

        // Fetch config for map bounds
        const config = await simConfig(session.session_id);

        // Create game engine
        gameEngine = new GameEngine(session.session_id, 'human', config);
        await gameEngine.start();

    } catch (err) {
        _startingSession = false;
        alert('Failed to start session: ' + err.message);
        navigateTo('landing');
    }
}


async function exitSession() {
    // Clear minimap canvas to remove stale pixels
    const minimapCanvas = document.getElementById('minimap-canvas');
    const mCtx = minimapCanvas.getContext('2d');
    mCtx.clearRect(0, 0, minimapCanvas.width, minimapCanvas.height);

    if (gameEngine) {
        gameEngine.destroy();
        gameEngine = null;
    }
    if (session && !_spectatorMode) {
        try {
            await deleteSession(session.session_id);
        } catch (_) {
            // ignore cleanup errors
        }
    }
    session = null;
    _spectatorMode = false;
    navigateTo('landing');
}

function setPlayStatus(status) {
    const dot = document.getElementById('play-status');
    dot.className = 'status-dot ' + (status || '');
}

// ===================================================================
// Spectator Mode (view a running session)
// ===================================================================

async function watchSession(sessionData) {
    try {
        _startingSession = true;
        _spectatorMode = true;
        navigateTo('play');
        setPlayStatus(sessionData.status);

        const scenarioData = scenarios.find(s => (s.scenario_id || s.id) === sessionData.scenario_id);
        document.getElementById('play-scenario-name').textContent =
            scenarioData ? (scenarioData.name || sessionData.scenario_id) : sessionData.scenario_id;
        document.getElementById('play-scenario-title').textContent =
            scenarioData ? (scenarioData.name || sessionData.scenario_id) : sessionData.scenario_id;
        document.getElementById('play-scenario-desc').textContent =
            scenarioData ? (scenarioData.description || '') : '';

        const badge = document.getElementById('play-mode-badge');
        badge.textContent = sessionData.mode === 'ai' ? 'AI' : 'HUMAN';
        badge.className = sessionData.mode === 'ai' ? 'badge badge-ai' : 'badge';

        // Show spectator badge, hide controls
        document.getElementById('play-spectator-badge').hidden = false;
        document.getElementById('controls-info').style.display = 'none';

        session = sessionData;
        _startingSession = false;

        // Fetch config for map bounds
        const config = await simConfig(session.session_id);

        // Create game engine in spectator mode (no controls, SSE only)
        gameEngine = new GameEngine(session.session_id, 'spectator', config);
        await gameEngine.start();

    } catch (err) {
        _startingSession = false;
        _spectatorMode = false;
        alert('Failed to watch session: ' + err.message);
        navigateTo('processes');
    }
}

// ===================================================================
// Active Processes
// ===================================================================

async function loadProcesses() {
    const container = document.getElementById('processes-list');
    try {
        // Also load scenarios for name resolution
        if (scenarios.length === 0) {
            scenarios = await fetchScenarios();
        }

        const data = await fetchSessions();
        const sessions = data.sessions || [];

        if (sessions.length === 0) {
            container.innerHTML = '<div class="empty-msg">No active sessions.</div>';
        } else {
            container.innerHTML = sessions.map(s => renderProcessCard(s)).join('');
        }

        // Start polling every 5s
        if (!_processesRefreshTimer) {
            _processesRefreshTimer = setInterval(loadProcesses, 5000);
        }
    } catch (err) {
        container.innerHTML = '<div class="empty-msg">Error: ' + escapeHtml(err.message) + '</div>';
    }
}

function renderProcessCard(s) {
    const scenarioData = scenarios.find(sc => (sc.scenario_id || sc.id) === s.scenario_id);
    const scenarioName = scenarioData ? (scenarioData.name || s.scenario_id) : s.scenario_id;
    const modeLabel = s.mode === 'ai' ? 'AI' : 'Human';
    const modeCls = s.mode === 'ai' ? 'badge badge-ai' : 'badge';
    const statusCls = 'status-dot ' + (s.status || '');
    const player = s.player_name || (s.model_info && s.model_info.model_name) || '-';
    const created = s.created_at ? new Date(s.created_at).toLocaleTimeString() : '';

    return `
        <div class="process-card">
            <div class="process-info">
                <h3>${escapeHtml(scenarioName)}</h3>
                <span class="${modeCls}">${modeLabel}</span>
                <span class="${statusCls}"></span>
                <span class="process-detail">Player: ${escapeHtml(player)}</span>
                <span class="process-detail">Started: ${escapeHtml(created)}</span>
                <span class="process-detail">Port: ${s.port}</span>
            </div>
            <div class="process-actions">
                <button class="btn btn-primary btn-sm" onclick="watchSessionById('${escapeAttr(s.session_id)}')">Watch</button>
                <button class="btn btn-ghost btn-sm" onclick="destroyProcessSession('${escapeAttr(s.session_id)}')">Stop</button>
            </div>
        </div>
    `;
}

async function watchSessionById(sessionId) {
    try {
        const sessionData = await apiFetch('/sessions/' + sessionId);
        await watchSession(sessionData);
    } catch (err) {
        alert('Session not found: ' + err.message);
    }
}

async function destroyProcessSession(sessionId) {
    if (!confirm('Stop this session?')) return;
    try {
        await deleteSession(sessionId);
        loadProcesses();
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

// ===================================================================
// Full Leaderboard View
// ===================================================================

// ===================================================================
// Results Page
// ===================================================================

let _resultsScenarioData = null; // cached scenario results for radar selection

function switchResultsTab(tab) {
    const tabs = ['global', 'scenario'];
    tabs.forEach(t => {
        document.getElementById('res-tab-' + t).classList.toggle('active', t === tab);
        document.getElementById('res-panel-' + t).hidden = t !== tab;
    });
    if (tab === 'global') {
        loadGlobalResults();
    } else {
        initScenarioResultsFilter();
    }
}

function setModeFilter(mode) {
    _currentModeFilter = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    // Reload whichever tab is active
    const globalActive = !document.getElementById('res-panel-global').hidden;
    if (globalActive) {
        loadGlobalResults();
    } else {
        const select = document.getElementById('res-filter-scenario');
        if (select && select.value) loadScenarioResults(select.value);
    }
}

async function loadResultsPage() {
    // Populate scenario filter
    if (scenarios.length === 0) {
        try { scenarios = await fetchScenarios(); } catch (_) {}
    }
    loadGlobalResults();
}

async function initScenarioResultsFilter() {
    if (scenarios.length === 0) {
        try { scenarios = await fetchScenarios(); } catch (_) {}
    }
    const select = document.getElementById('res-filter-scenario');
    if (select.options.length === 0) {
        scenarios.forEach(s => {
            const id = s.scenario_id || s.id;
            const name = s.name || id;
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = name;
            select.appendChild(opt);
        });
        select.onchange = () => loadScenarioResults(select.value);
    }
    if (select.value) {
        loadScenarioResults(select.value);
    }
}

async function loadGlobalResults() {
    const tableContainer = document.getElementById('global-results-table');
    const parallelContainer = document.getElementById('global-parallel-plot');
    const radarContainer = document.getElementById('global-radar-chart');
    tableContainer.innerHTML = '<div class="loading-msg">Loading...</div>';
    parallelContainer.innerHTML = '';
    radarContainer.innerHTML = '';

    try {
        const modeParam = _currentModeFilter ? '?mode=' + _currentModeFilter : '';
        const data = await apiFetch('/leaderboard/global' + modeParam);
        const agents = data.agents || [];

        if (agents.length === 0) {
            tableContainer.innerHTML = '<div class="empty-msg">No agents have completed all scenarios yet.</div>';
            return;
        }

        // Store for reuse
        _resultsGlobalData = agents;

        // Compute normalized scores across all agents
        _addNormalizedScores(agents, ['avg_steps_taken', 'avg_duration_s', 'avg_distance_travelled']);

        // Build sortable table with checkboxes
        const columns = [
            { key: 'pareto_rank',  label: 'Rank',          sortable: true,  format: v => v },
            { key: 'agent_name',   label: 'Agent',         sortable: true,  format: null },
            { key: 'success_rate', label: 'Success Rate',   sortable: true,  format: v => (v * 100).toFixed(0) + '%', best: 'max' },
            { key: 'avg_steps_taken',      label: 'Avg Steps',  sortable: true,  format: v => v != null ? Math.round(v) : '-', best: 'min' },
            { key: 'avg_steps_taken_score', label: 'Steps Score', sortable: true, format: v => v != null ? v.toFixed(2) : '-', best: 'max' },
            { key: 'avg_duration_s',       label: 'Avg Time (s)', sortable: true, format: v => v != null ? v.toFixed(1) : '-', best: 'min' },
            { key: 'avg_duration_s_score', label: 'Time Score', sortable: true, format: v => v != null ? v.toFixed(2) : '-', best: 'max' },
            { key: 'avg_distance_travelled', label: 'Avg Dist (m)', sortable: true, format: v => v != null ? Math.round(v) : '-', best: 'min' },
            { key: 'avg_distance_travelled_score', label: 'Dist Score', sortable: true, format: v => v != null ? v.toFixed(2) : '-', best: 'max' },
            { key: '_check',       label: 'Plot',           sortable: false, format: null },
        ];

        _renderResultsTable(tableContainer, agents, columns, 'global');

        // Plots (top 5 selected by default)
        _updateResultsPlots('global');

    } catch (err) {
        tableContainer.innerHTML = '<div class="empty-msg">Error: ' + escapeHtml(err.message) + '</div>';
    }
}

async function loadScenarioResults(scenarioId) {
    if (!scenarioId) return;
    const tableContainer = document.getElementById('scenario-results-table');
    const parallelContainer = document.getElementById('scenario-parallel-plot');
    const radarContainer = document.getElementById('scenario-radar-chart');
    const selectorContainer = document.getElementById('radar-agent-selector');
    tableContainer.innerHTML = '<div class="loading-msg">Loading...</div>';
    parallelContainer.innerHTML = '';
    radarContainer.innerHTML = '';
    if (selectorContainer) selectorContainer.innerHTML = '';

    try {
        const modeParam = _currentModeFilter ? '?mode=' + _currentModeFilter : '';
        const data = await apiFetch('/leaderboard/scenario/' + encodeURIComponent(scenarioId) + modeParam);
        const agents = data.agents || [];
        _resultsScenarioData = agents;

        if (agents.length === 0) {
            tableContainer.innerHTML = '<div class="empty-msg">No runs for this scenario yet.</div>';
            return;
        }

        // Compute normalized scores across all agents
        _addNormalizedScores(agents, ['steps_taken', 'duration_s', 'distance_travelled']);

        const columns = [
            { key: 'pareto_rank',  label: 'Rank',          sortable: true,  format: v => v },
            { key: 'agent_name',   label: 'Agent',         sortable: true,  format: null },
            { key: 'success',      label: 'Result',        sortable: true,  format: null },
            { key: 'steps_taken',  label: 'Steps',         sortable: true,  format: v => v != null ? v : '-', best: 'min' },
            { key: 'steps_taken_score', label: 'Steps Score', sortable: true, format: v => v != null ? v.toFixed(2) : '-', best: 'max' },
            { key: 'duration_s',   label: 'Time (s)',      sortable: true,  format: v => v != null ? v.toFixed(1) : '-', best: 'min' },
            { key: 'duration_s_score', label: 'Time Score', sortable: true, format: v => v != null ? v.toFixed(2) : '-', best: 'max' },
            { key: 'distance_travelled', label: 'Distance (m)', sortable: true, format: v => v != null ? Math.round(v) : '-', best: 'min' },
            { key: 'distance_travelled_score', label: 'Dist Score', sortable: true, format: v => v != null ? v.toFixed(2) : '-', best: 'max' },
            { key: 'target_seen',  label: 'Target Seen',   sortable: true,  format: v => v ? 'Yes' : 'No', best: 'max' },
            { key: '_check',       label: 'Plot',          sortable: false },
        ];

        _renderResultsTable(tableContainer, agents, columns, 'scenario');
        _updateResultsPlots('scenario');

    } catch (err) {
        tableContainer.innerHTML = '<div class="empty-msg">Error: ' + escapeHtml(err.message) + '</div>';
    }
}

// ===================================================================
// Sortable Results Table with Checkboxes
// ===================================================================

const AGENT_COLORS = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16', '#06b6d4', '#e11d48'];

let _resultsGlobalData = [];
// Sort state per scope: null = default (pareto_rank asc), else {key, asc}
let _globalSort = null;
let _scenarioSort = null;

function _renderResultsTable(container, agents, columns, scope) {
    const sortState = scope === 'global' ? _globalSort : _scenarioSort;
    const sortKey = sortState ? sortState.key : 'pareto_rank';
    const sortAsc = sortState ? sortState.asc : true;

    // Sort agents
    const sorted = [...agents].sort((a, b) => {
        let va = a[sortKey], vb = b[sortKey];
        if (va == null) va = sortAsc ? Infinity : -Infinity;
        if (vb == null) vb = sortAsc ? Infinity : -Infinity;
        if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        if (typeof va === 'boolean') { va = va ? 1 : 0; vb = vb ? 1 : 0; }
        return sortAsc ? va - vb : vb - va;
    });

    // Find best values per column
    const bestVals = {};
    columns.forEach(col => {
        if (!col.best) return;
        const vals = agents.map(a => a[col.key]).filter(v => v != null && v !== false);
        if (vals.length === 0) return;
        bestVals[col.key] = col.best === 'max' ? Math.max(...vals) : Math.min(...vals);
    });

    // Determine which agents are checked (preserve previous selection or default top 5)
    const prevChecked = _getCheckedAgents(scope);
    const checkedSet = prevChecked.length > 0
        ? new Set(prevChecked)
        : new Set(agents.slice(0, 5).map(a => a.agent_name));

    let html = '<table class="leaderboard-table sortable-table"><thead><tr>';
    columns.forEach(col => {
        if (col.key === '_check') {
            html += '<th class="col-check">Plot</th>';
        } else if (col.sortable) {
            let arrow = '';
            if (sortState && sortState.key === col.key) {
                arrow = sortState.asc ? ' ▲' : ' ▼';
            }
            html += '<th class="sortable-th" data-sort-key="' + col.key + '" data-scope="' + scope + '">' + col.label + arrow + '</th>';
        } else {
            html += '<th>' + col.label + '</th>';
        }
    });
    html += '</tr></thead><tbody>';

    sorted.forEach((a) => {
        const colorIdx = agents.indexOf(a);
        const color = AGENT_COLORS[colorIdx % AGENT_COLORS.length];
        const checked = checkedSet.has(a.agent_name) ? 'checked' : '';

        html += '<tr>';
        columns.forEach(col => {
            if (col.key === '_check') {
                html += '<td class="col-check">'
                    + '<input type="checkbox" class="agent-plot-check" data-scope="' + scope + '" data-agent="' + escapeAttr(a.agent_name) + '" ' + checked
                    + ' onchange="_onAgentCheckChange(\'' + scope + '\')">'
                    + '<span class="agent-color-dot" style="background:' + color + '"></span></td>';
            } else if (col.key === 'agent_name') {
                const link = '<a href="#agent/' + encodeURIComponent(a.agent_name) + '" class="agent-link">' + escapeHtml(a.agent_name) + '</a>';
                html += '<td>' + link + '</td>';
            } else if (col.key === 'success') {
                const cls = a.success ? 'result-success' : 'result-fail';
                const txt = a.success ? 'SUCCESS' : 'FAIL';
                html += '<td><span class="' + cls + '">' + txt + '</span></td>';
            } else {
                const val = a[col.key];
                const display = col.format ? col.format(val) : val;
                const isBest = col.best && val != null && val === bestVals[col.key];
                const cls = isBest ? ' class="best-val"' : '';
                html += '<td' + cls + '>' + display + '</td>';
            }
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;

    // Attach 3-state sort handlers: asc → desc → default
    container.querySelectorAll('.sortable-th').forEach(th => {
        th.addEventListener('click', () => {
            const key = th.dataset.sortKey;
            const sc = th.dataset.scope;
            const cur = sc === 'global' ? _globalSort : _scenarioSort;
            let next;
            if (!cur || cur.key !== key) {
                next = { key, asc: true };
            } else if (cur.asc) {
                next = { key, asc: false };
            } else {
                next = null; // back to default
            }
            if (sc === 'global') { _globalSort = next; }
            else { _scenarioSort = next; }
            _renderResultsTable(container, sc === 'global' ? _resultsGlobalData : _resultsScenarioData, columns, sc);
            _updateResultsPlots(sc);
        });
    });
}

function _getCheckedAgents(scope) {
    const checks = document.querySelectorAll('.agent-plot-check[data-scope="' + scope + '"]:checked');
    return Array.from(checks).map(cb => cb.dataset.agent);
}

function _onAgentCheckChange(scope) {
    _updateResultsPlots(scope);
}

/**
 * Compute min-max normalized scores for lower-is-better metrics.
 * Score = 1 - (val - min) / (max - min).  Best agent gets 1.0, worst 0.0.
 * Adds *_score properties to each object in-place.
 */
function _addNormalizedScores(items, metricKeys) {
    if (!items || items.length === 0) return;
    metricKeys.forEach(key => {
        const scoreKey = key + '_score';
        const vals = items.map(a => a[key]).filter(v => v != null);
        if (vals.length === 0) { items.forEach(a => { a[scoreKey] = null; }); return; }
        const mn = Math.min(...vals), mx = Math.max(...vals);
        const span = mx - mn;
        items.forEach(a => {
            const v = a[key];
            if (v == null) { a[scoreKey] = null; return; }
            a[scoreKey] = span > 0 ? 1 - (v - mn) / span : 1.0;
        });
    });
}

function _updateResultsPlots(scope) {
    const allData = scope === 'global' ? _resultsGlobalData : _resultsScenarioData;
    const selected = _getCheckedAgents(scope);
    const isGlobal = scope === 'global';

    const parallelContainer = document.getElementById(scope + '-parallel-plot');
    const radarContainer = document.getElementById(scope + '-radar-chart');

    if (!selected || selected.length === 0) {
        parallelContainer.innerHTML = '<div class="empty-msg">Select agents in the table to compare</div>';
        radarContainer.innerHTML = '<div class="empty-msg">Select agents in the table to compare</div>';
        return;
    }

    const filtered = allData.filter(a => selected.includes(a.agent_name));
    const mapped = filtered.map(a => ({
        agent_name: a.agent_name,
        success_rate: isGlobal ? a.success_rate : (a.success ? 1 : 0),
        steps_taken: isGlobal ? a.avg_steps_taken : a.steps_taken,
        duration_s: isGlobal ? a.avg_duration_s : a.duration_s,
        distance_travelled: isGlobal ? a.avg_distance_travelled : a.distance_travelled,
        steps_score: isGlobal ? a.avg_steps_taken_score : a.steps_taken_score,
        time_score: isGlobal ? a.avg_duration_s_score : a.duration_s_score,
        dist_score: isGlobal ? a.avg_distance_travelled_score : a.distance_travelled_score,
        _colorIdx: allData.indexOf(a),
    }));

    renderParallelPlot(parallelContainer, mapped);
    renderRadarChart(radarContainer, mapped);
}

function updateScenarioRadar() {
    _updateResultsPlots('scenario');
}

// ===================================================================
// Plotly Visualisations
// ===================================================================

function renderParallelPlot(container, agents) {
    if (!agents || agents.length === 0 || typeof Plotly === 'undefined') {
        container.innerHTML = '';
        return;
    }
    const plotWidth = container.clientWidth || 500;

    const dims = ['steps_score', 'time_score', 'dist_score'];
    const dimLabels = ['Steps Score', 'Time Score', 'Dist Score'];

    // Values are already normalized [0, 1] — higher is better
    const traces = agents.map((a, i) => {
        const color = AGENT_COLORS[(a._colorIdx != null ? a._colorIdx : i) % AGENT_COLORS.length];
        const vals = dims.map(d => a[d] || 0);
        const hoverText = dims.map((d, di) => {
            const v = a[d];
            return dimLabels[di] + ': ' + (v != null ? v.toFixed(2) : '-');
        }).join('<br>');
        return {
            type: 'scatter',
            mode: 'lines+markers',
            x: dimLabels,
            y: vals,
            name: a.agent_name,
            line: { color: color, width: 2.5 },
            marker: { color: color, size: 7 },
            hovertemplate: hoverText + '<extra>' + a.agent_name + '</extra>',
        };
    });

    const layout = {
        title: { text: 'Parallel Coordinates', font: { color: '#e2e8f0', size: 14 }, x: 0.5, y: 0.98 },
        margin: { l: 30, r: 30, t: 45, b: 70 },
        font: { color: '#a0aec0', size: 11 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        width: plotWidth,
        height: 420,
        xaxis: { gridcolor: '#2d3748', linecolor: '#2d3748', fixedrange: true },
        yaxis: { range: [-0.05, 1.05], showticklabels: true, gridcolor: '#2d3748', linecolor: '#2d3748', fixedrange: true, zeroline: false, tickformat: '.1f' },
        legend: { font: { color: '#a0aec0', size: 10 }, orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center' },
        hovermode: 'closest',
    };

    Plotly.newPlot(container, traces, layout, { responsive: false, displayModeBar: false });
}

function renderRadarChart(container, agents) {
    if (!agents || agents.length === 0 || typeof Plotly === 'undefined') {
        container.innerHTML = '';
        return;
    }
    const plotWidth = container.clientWidth || 500;

    const categories = ['Steps Score', 'Time Score', 'Dist Score'];

    const traces = agents.map((a, i) => {
        const color = AGENT_COLORS[(a._colorIdx != null ? a._colorIdx : i) % AGENT_COLORS.length];
        const r = [
            a.steps_score || 0,
            a.time_score || 0,
            a.dist_score || 0,
        ];
        return {
            type: 'scatterpolar',
            r: [...r, r[0]],
            theta: [...categories, categories[0]],
            fill: 'toself',
            fillcolor: color + '20',
            line: { color: color, width: 2 },
            name: a.agent_name,
            opacity: 0.8,
        };
    });

    const layout = {
        title: { text: 'Radar Comparison', font: { color: '#e2e8f0', size: 14 }, x: 0.5, y: 0.98 },
        polar: {
            radialaxis: { visible: true, range: [0, 1], showticklabels: false, gridcolor: '#2d3748' },
            angularaxis: { gridcolor: '#2d3748', linecolor: '#2d3748', rotation: 90 },
            bgcolor: 'transparent',
            domain: { x: [0.1, 0.9], y: [0.05, 0.85] },
        },
        showlegend: true,
        legend: { font: { color: '#a0aec0', size: 10 }, orientation: 'h', y: -0.1, x: 0.5, xanchor: 'center' },
        margin: { l: 40, r: 40, t: 45, b: 60 },
        font: { color: '#a0aec0', size: 11 },
        paper_bgcolor: 'transparent',
        width: plotWidth,
        height: 420,
    };

    Plotly.newPlot(container, traces, layout, { responsive: false, displayModeBar: false });
}

// ===================================================================
// Game Engine
// ===================================================================

class GameEngine {
    constructor(sessionId, mode, config) {
        this.sessionId = sessionId;
        this.mode = mode;
        this.config = config;
        this.renderer = new CanvasRenderer(document.getElementById('main-canvas'));
        this.minimapRenderer = new MinimapRenderer(
            document.getElementById('minimap-canvas'),
            config.map_bounds
        );
        this.controls = null;
        this.clickHandler = null;
        this.done = false;
        this.step = 0;
        this.showHud = true;
        this.showMinimap = true;
        this.currentModality = null;
        this.modalityList = [];
        this.modalityIndex = 0;
        this._destroyed = false;
        this._processing = false; // debounce steps
        this._sseSource = null;   // EventSource for SSE
        this._frameQueue = [];    // queued SSE observations for smooth playback
        this._frameTimer = null;  // interval id for frame playback
    }

    async start() {
        trajectory = [];
        stepsDetail = [];
        startTime = Date.now();
        setPlayStatus('running');

        // Clear minimap at start
        this.minimapRenderer.reset();
        const mCanvas = document.getElementById('minimap-canvas');
        const mCtx = mCanvas.getContext('2d');
        mCtx.clearRect(0, 0, mCanvas.width, mCanvas.height);

        if (this.mode === 'spectator') {
            // Spectator: don't reset, don't create controls.
            // Fetch current snapshot + telemetry trail, then stream SSE.
            await this._initSpectatorState();
            this._connectSSE();
            return;
        }

        // Reset episode (human / AI modes)
        const obs = await simReset(this.sessionId);
        this._handleObservation(obs);

        // Setup controls for human mode
        if (this.mode === 'human') {
            // Open SSE BEFORE enabling controls so smooth micro-step
            // frames are received as soon as the first action fires.
            await this._connectSSE();
            this.controls = new HumanControls(this);
            this._setupClickToMove();
        }
    }

    _connectSSE() {
        const url = API_BASE + '/sessions/' + this.sessionId + '/sim/events';
        const source = new EventSource(url);
        this._sseSource = source;

        source.addEventListener('observation', (event) => {
            if (this._destroyed) {
                source.close();
                return;
            }
            try {
                const obs = JSON.parse(event.data);
                // Queue the frame for smooth playback instead of
                // rendering immediately (avoids burst rendering when
                // multiple SSE events arrive in the same JS tick).
                this._frameQueue.push(obs);
                this._startFramePlayback();
            } catch (err) {
                console.error('SSE parse error:', err);
            }
        });

        source.onerror = () => {
            if (this._destroyed) return;
            setPlayStatus('');
            console.warn('SSE connection lost');
        };

        // Return a promise that resolves once the connection is open,
        // so callers can await readiness before enabling controls.
        return new Promise((resolve) => {
            if (source.readyState === EventSource.OPEN) {
                resolve();
            } else {
                source.addEventListener('open', () => resolve(), { once: true });
                setTimeout(resolve, 2000);
            }
        });
    }

    _startFramePlayback() {
        // If a playback timer is already running, let it drain the queue.
        if (this._frameTimer) return;
        // Render queued frames one at a time with a minimum interval
        // so each frame is actually visible on screen (~20 FPS).
        const FRAME_INTERVAL = 50; // ms
        this._frameTimer = setInterval(() => {
            if (this._destroyed || this._frameQueue.length === 0) {
                clearInterval(this._frameTimer);
                this._frameTimer = null;
                return;
            }
            const obs = this._frameQueue.shift();
            this._handleObservation(obs);
        }, FRAME_INTERVAL);
        // Also render the first frame immediately so there's no initial
        // delay when a single frame arrives (e.g. spectator idle).
        if (this._frameQueue.length > 0) {
            const first = this._frameQueue.shift();
            this._handleObservation(first);
        }
    }

    async _initSpectatorState() {
        // Fetch the last snapshot (full observation) and telemetry trail
        // so the spectator sees the current cockpit view + minimap history
        // immediately, without waiting for the next SSE event.
        const prefix = '/sessions/' + this.sessionId + '/sim';

        // Fire both requests in parallel.
        const [snapshotResult, telemetryResult] = await Promise.allSettled([
            apiFetch(prefix + '/snapshot'),
            apiFetch(prefix + '/telemetry'),
        ]);

        // Reconstruct minimap trail from telemetry records.
        if (telemetryResult.status === 'fulfilled' && telemetryResult.value) {
            const telem = telemetryResult.value;
            const records = telem.records || [];
            for (const r of records) {
                this.minimapRenderer.addPoint(Number(r.x), Number(r.y));
            }
        }

        // Render last snapshot (image + telemetry panel + minimap position).
        if (snapshotResult.status === 'fulfilled' && snapshotResult.value) {
            this._handleObservation(snapshotResult.value);
        } else {
            // No snapshot yet — try fetching at least the drone state
            // for a minimap dot.
            try {
                const st = await apiFetch(prefix + '/state');
                if (st) {
                    this.minimapRenderer.addPoint(st.x || 0, st.y || 0);
                    this.minimapRenderer.render(st.x || 0, st.y || 0);
                }
            } catch (_) {
                // Session may not have an active episode yet
            }
        }
    }

    _setupClickToMove() {
        const canvas = document.getElementById('main-canvas');
        this.clickHandler = (e) => {
            if (this.done || this._destroyed || this._processing) return;
            if (!lastObservation) return;

            const rect = canvas.getBoundingClientRect();

            // Click position relative to canvas (0..1 normalized)
            const clickX = (e.clientX - rect.left) / rect.width;
            const clickY = (e.clientY - rect.top) / rect.height;

            // Reject clicks outside the canvas bounds
            if (clickX < 0 || clickX > 1 || clickY < 0 || clickY > 1) return;

            // The camera captures a square ground patch.
            // At current altitude z with FOV 90deg: half_extent = z * tan(45) = z
            // Ground patch is centered on (drone.x, drone.y), spanning:
            //   x: [drone.x - half_extent, drone.x + half_extent]
            //   y: [drone.y - half_extent, drone.y + half_extent]
            //
            // The image pixel (0,0) = top-left = (x_min, y_max) in world coords
            // Canvas (0,0) is top-left.
            //   clickX=0 → world x_min = drone.x - half_extent
            //   clickX=1 → world x_max = drone.x + half_extent
            //   clickY=0 → world y_max (top of image = north)
            //   clickY=1 → world y_min (bottom of image = south)

            const ds = lastObservation.drone_state || {};
            const droneX = ds.x || 0;
            const droneY = ds.y || 0;
            const droneZ = ds.z || 100;

            // half_extent = z * tan(fov/2). Default FOV=90deg → tan(45)=1
            const halfExtent = droneZ; // Assumes 90deg FOV

            const worldX = droneX - halfExtent + clickX * 2 * halfExtent;
            const worldY = droneY + halfExtent - clickY * 2 * halfExtent;

            const dx = worldX - droneX;
            const dy = worldY - droneY;

            if (Math.abs(dx) < 1 && Math.abs(dy) < 1) return; // too close, ignore

            this.step_({ dx: dx, dy: dy, dz: 0, action_type: 'move' });
        };
        canvas.addEventListener('click', this.clickHandler);
    }

    async step_(action) {
        if (this.done || this._destroyed || this._processing) return;
        this._processing = true;
        try {
            const obs = await simStep(this.sessionId, action);
            // Record step detail (action metadata, not rendering).
            stepsDetail.push({
                dx: action.dx,
                dy: action.dy,
                dz: action.dz,
                action_type: action.action_type,
                reason: action.reason || null,
            });
            // When SSE is connected, rendering is handled by the SSE
            // listener (_handleObservation from SSE events).  The HTTP
            // response is the final frame of a potentially smooth
            // movement — the same frame SSE will deliver.
            //
            // _processing is released by _handleObservation when it
            // detects a logical step increment (obs.step > this.step).
            //
            // Fallback: if SSE is not connected (should not happen in
            // human mode, but be safe), render from HTTP and release.
            if (!this._sseSource || this._sseSource.readyState === EventSource.CLOSED) {
                this._handleObservation(obs);
                this._processing = false;
            }
        } catch (err) {
            console.error('Step error:', err);
            this._processing = false;
        }
    }

    _handleObservation(obs) {
        if (this._destroyed) return;
        lastObservation = obs;

        const ds = obs.drone_state || {};
        const obsStep = obs.step || 0;

        // Detect logical step increment (as opposed to intermediate micro-step).
        // Micro-steps have the same step count as the previous observation.
        // Also treat the first observation (reset) as a logical step.
        const isLogicalStep = obsStep > this.step || trajectory.length === 0;
        if (isLogicalStep) {
            this.step = obsStep;
            // Record trajectory only for logical steps (not micro-steps).
            trajectory.push({
                x: ds.x || 0,
                y: ds.y || 0,
                z: ds.z || 0,
                step: this.step,
                footprint: 2 * (ds.z || 0),  // ground coverage at FOV=90deg: width = 2*altitude
            });
            // Release processing lock — the smooth movement is complete.
            this._processing = false;
        }

        // Discover modalities
        if (obs.images && Object.keys(obs.images).length > 0) {
            this.modalityList = Object.keys(obs.images).sort();
            if (!this.currentModality || !this.modalityList.includes(this.currentModality)) {
                this.currentModality = this.modalityList[0];
                this.modalityIndex = 0;
            }
        }

        // Render main canvas (no HUD overlay -- telemetry is in HTML panel)
        const imageB64 = this.currentModality && obs.images && obs.images[this.currentModality]
            ? obs.images[this.currentModality]
            : obs.image_base64;

        this.renderer.drawObservation(imageB64);

        // Update telemetry panel
        this._updateTelemetry(obs);

        // Minimap — show ALL frames (smooth trail including micro-steps)
        if (this.showMinimap) {
            this.minimapRenderer.addPoint(ds.x || 0, ds.y || 0);
            // Only show target in AI mode -- hide in human mode to avoid cheating
            if (this.mode !== 'human') {
                const meta = obs.metadata || {};
                if (meta.target_x != null && meta.target_y != null) {
                    this.minimapRenderer.setTarget(
                        Number(meta.target_x), Number(meta.target_y)
                    );
                }
            }
            this.minimapRenderer.render(ds.x || 0, ds.y || 0);
        }

        // Check done — guard with !this.done to prevent duplicate modals
        // (SSE + HTTP may both deliver the final observation).
        if (obs.result && !this.done) {
            this.done = true;
            this._processing = false;
            setPlayStatus('finished');
            // Only show results modal for human players (not spectators, not AI).
            // AI agents submit results via the API endpoint directly.
            if (this.mode === 'human') {
                const elapsed = startTime ? ((Date.now() - startTime) / 1000) : 0;
                showResultsModal(obs, trajectory, elapsed);
            }
        }
    }

    _updateTelemetry(obs) {
        const ds = obs.drone_state || {};
        const elapsed = startTime ? ((Date.now() - startTime) / 1000) : 0;
        const maxSteps = this.config.max_steps;

        document.getElementById('telem-step').textContent =
            this.step + (maxSteps ? '/' + maxSteps : '');
        document.getElementById('telem-x').textContent = Math.round(ds.x || 0);
        document.getElementById('telem-y').textContent = Math.round(ds.y || 0);
        document.getElementById('telem-alt').textContent =
            Math.round(ds.z || 0) + ' m';
        document.getElementById('telem-dist').textContent =
            Math.round(ds.total_distance || 0) + ' m';
        document.getElementById('telem-time').textContent = formatDuration(elapsed);

        // Modality row
        const modRow = document.getElementById('telem-modality-row');
        if (this.modalityList.length > 1) {
            modRow.hidden = false;
            document.getElementById('telem-modality').textContent =
                this.currentModality + ' (' + (this.modalityIndex + 1) + '/' + this.modalityList.length + ')';
        } else {
            modRow.hidden = true;
        }

        // Status line
        const statusEl = document.getElementById('telem-status');
        if (obs.result) {
            if (obs.result.success) {
                statusEl.textContent = 'SUCCESS';
                statusEl.style.color = 'var(--success)';
            } else {
                statusEl.textContent = 'FAILED: ' + (obs.result.reason || '');
                statusEl.style.color = 'var(--fail)';
            }
        } else {
            statusEl.textContent = '';
        }
    }

    cycleModality() {
        if (this.modalityList.length <= 1) return;
        this.modalityIndex = (this.modalityIndex + 1) % this.modalityList.length;
        this.currentModality = this.modalityList[this.modalityIndex];
        // Re-render with new modality if we have a last observation
        if (lastObservation) {
            this._handleObservation(lastObservation);
        }
    }

    toggleHud() {
        this.showHud = !this.showHud;
        document.getElementById('telemetry-panel').style.display =
            this.showHud ? '' : 'none';
    }

    toggleMinimap() {
        this.showMinimap = !this.showMinimap;
        const mc = document.querySelector('.minimap-container');
        mc.style.display = this.showMinimap ? '' : 'none';
    }

    async resetEpisode() {
        this.done = false;
        this.step = 0;
        trajectory = [];
        stepsDetail = [];
        startTime = Date.now();
        this.minimapRenderer.reset();
        // Clear minimap canvas
        const mCanvas = document.getElementById('minimap-canvas');
        const mCtx = mCanvas.getContext('2d');
        mCtx.clearRect(0, 0, mCanvas.width, mCanvas.height);
        setPlayStatus('running');
        document.getElementById('results-modal').hidden = true;
        const obs = await simReset(this.sessionId);
        this._handleObservation(obs);
    }

    destroy() {
        this._destroyed = true;
        if (this._frameTimer) {
            clearInterval(this._frameTimer);
            this._frameTimer = null;
        }
        this._frameQueue = [];
        if (this._sseSource) {
            this._sseSource.close();
            this._sseSource = null;
        }
        if (this.controls) {
            this.controls.destroy();
            this.controls = null;
        }
        if (this.clickHandler) {
            const canvas = document.getElementById('main-canvas');
            canvas.removeEventListener('click', this.clickHandler);
            this.clickHandler = null;
        }
    }
}

// ===================================================================
// Canvas Renderer  (FIXED: never change canvas.width/height)
// ===================================================================

class CanvasRenderer {
    constructor(canvasEl) {
        this.canvas = canvasEl;
        this.ctx = canvasEl.getContext('2d');
        // Canvas is fixed at 500x500 — never change dimensions
    }

    drawObservation(imageBase64) {
        if (!imageBase64) {
            this._drawWaiting('No image data');
            return;
        }
        const img = new Image();
        const ctx = this.ctx;
        const canvas = this.canvas;

        img.onload = () => {
            // Draw image scaled to fit canvas -- never resize canvas
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            // Draw drone center reticle
            this._drawReticle(ctx, canvas.width / 2, canvas.height / 2);
        };
        img.src = 'data:image/png;base64,' + imageBase64;
    }

    _drawReticle(ctx, cx, cy) {
        const r = 8;   // circle radius
        const g = 4;   // gap from center
        const len = 14; // line length from center
        ctx.save();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = 1.5;
        // Circle
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
        // Cross lines (with gap around center)
        ctx.beginPath();
        ctx.moveTo(cx - len, cy); ctx.lineTo(cx - g, cy);
        ctx.moveTo(cx + g, cy);   ctx.lineTo(cx + len, cy);
        ctx.moveTo(cx, cy - len); ctx.lineTo(cx, cy - g);
        ctx.moveTo(cx, cy + len); ctx.lineTo(cx, cy + g);
        ctx.stroke();
        ctx.restore();
    }

    _drawWaiting(msg) {
        const ctx = this.ctx;
        ctx.fillStyle = '#1e1e1e';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.fillStyle = '#c8c8c8';
        ctx.font = '20px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(msg, this.canvas.width / 2, this.canvas.height / 2);
        ctx.textAlign = 'start';
    }
}

// ===================================================================
// Minimap Renderer
// ===================================================================

class MinimapRenderer {
    constructor(canvasEl, mapBounds) {
        this.canvas = canvasEl;
        this.ctx = canvasEl.getContext('2d');
        this.bounds = mapBounds || { x_min: 0, y_min: 0, x_max: 1000, y_max: 1000 };
        this.trail = [];
        this.targetX = null;
        this.targetY = null;
        this.size = canvasEl.width; // assumes square
    }

    reset() {
        this.trail = [];
        this.targetX = null;
        this.targetY = null;
        // Explicit clear to avoid stale pixels
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    addPoint(x, y) {
        // Deduplicate: skip if same as last point
        if (this.trail.length > 0) {
            const last = this.trail[this.trail.length - 1];
            if (Math.abs(last.x - x) < 0.01 && Math.abs(last.y - y) < 0.01) return;
        }
        this.trail.push({ x, y });
    }

    setTarget(x, y) {
        this.targetX = x;
        this.targetY = y;
    }

    _worldToMinimap(wx, wy) {
        const b = this.bounds;
        const w = (b.x_max || 1) - (b.x_min || 0);
        const h = (b.y_max || 1) - (b.y_min || 0);
        const marginFrac = 0.05;
        const usable = 1.0 - 2 * marginFrac;

        const nx = w > 0 ? (wx - b.x_min) / w : 0.5;
        const ny = h > 0 ? 1.0 - (wy - b.y_min) / h : 0.5; // flip Y

        return {
            x: (marginFrac + nx * usable) * this.size,
            y: (marginFrac + ny * usable) * this.size,
        };
    }

    render(droneX, droneY) {
        const ctx = this.ctx;
        const s = this.size;

        // Background
        ctx.fillStyle = 'rgba(20, 20, 20, 0.784)';
        ctx.fillRect(0, 0, s, s);

        // Border
        ctx.strokeStyle = '#505050';
        ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, s - 1, s - 1);

        // Trail
        if (this.trail.length >= 2) {
            ctx.beginPath();
            ctx.strokeStyle = '#64C8FF';
            ctx.lineWidth = 1;
            const p0 = this._worldToMinimap(this.trail[0].x, this.trail[0].y);
            ctx.moveTo(p0.x, p0.y);
            for (let i = 1; i < this.trail.length; i++) {
                const p = this._worldToMinimap(this.trail[i].x, this.trail[i].y);
                ctx.lineTo(p.x, p.y);
            }
            ctx.stroke();
        }

        // Drone
        const dp = this._worldToMinimap(droneX, droneY);
        ctx.beginPath();
        ctx.fillStyle = '#FF3232';
        ctx.arc(dp.x, dp.y, 4, 0, Math.PI * 2);
        ctx.fill();

        // Target crosshair
        if (this.targetX != null && this.targetY != null) {
            const tp = this._worldToMinimap(this.targetX, this.targetY);
            const r = 5;
            ctx.strokeStyle = '#32FF32';
            ctx.lineWidth = 1;

            // Circle
            ctx.beginPath();
            ctx.arc(tp.x, tp.y, r, 0, Math.PI * 2);
            ctx.stroke();

            // Crosshair lines
            ctx.beginPath();
            ctx.moveTo(tp.x - r - 2, tp.y);
            ctx.lineTo(tp.x + r + 2, tp.y);
            ctx.moveTo(tp.x, tp.y - r - 2);
            ctx.lineTo(tp.x, tp.y + r + 2);
            ctx.stroke();
        }
    }
}

// ===================================================================
// Human Controls
// ===================================================================

class HumanControls {
    constructor(engine) {
        this.engine = engine;
        this.moveStep = 20;
        this.altStep = 10;
        this._boundKeyDown = this._onKeyDown.bind(this);
        document.addEventListener('keydown', this._boundKeyDown);

        // Step size slider
        const slider = document.getElementById('step-size-slider');
        const label = document.getElementById('step-size-value');
        slider.value = this.moveStep;
        label.textContent = this.moveStep;
        this._sliderHandler = () => {
            this.moveStep = parseInt(slider.value, 10);
            label.textContent = this.moveStep;
        };
        slider.addEventListener('input', this._sliderHandler);
    }

    _onKeyDown(e) {
        if (this.engine.done || this.engine._destroyed) return;

        // Ignore if typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        let action = null;

        switch (e.key) {
            // Movement -- ZQSD (AZERTY) and WASD (QWERTY) + arrows
            case 'z': case 'Z': case 'w': case 'W': case 'ArrowUp':
                action = { dx: 0, dy: this.moveStep, dz: 0, action_type: 'move' };
                break;
            case 's': case 'S': case 'ArrowDown':
                action = { dx: 0, dy: -this.moveStep, dz: 0, action_type: 'move' };
                break;
            case 'd': case 'D': case 'ArrowRight':
                action = { dx: this.moveStep, dy: 0, dz: 0, action_type: 'move' };
                break;
            case 'q': case 'Q': case 'ArrowLeft':
                action = { dx: -this.moveStep, dy: 0, dz: 0, action_type: 'move' };
                break;

            // Altitude
            case 'e': case 'E':
                action = { dx: 0, dy: 0, dz: this.altStep, action_type: 'move' };
                break;
            case 'a': case 'A':
                action = { dx: 0, dy: 0, dz: -this.altStep, action_type: 'move' };
                break;

            // Step size adjust
            case '+': case '=':
                this.moveStep = Math.min(500, Math.round(this.moveStep * 1.5));
                this._updateSlider();
                return;
            case '-':
                this.moveStep = Math.max(1, Math.round(this.moveStep / 1.5));
                this._updateSlider();
                return;

            // Actions
            case ' ':
                e.preventDefault();
                action = { dx: 0, dy: 0, dz: 0, action_type: 'found' };
                break;
            case 'Escape':
                action = { dx: 0, dy: 0, dz: 0, action_type: 'stop' };
                break;

            // Toggles
            case 'r': case 'R':
                e.preventDefault();
                this.engine.resetEpisode();
                return;
            case 'h': case 'H':
                this.engine.toggleHud();
                return;
            case 'm': case 'M':
                this.engine.toggleMinimap();
                return;
            case 'Tab':
                e.preventDefault();
                this.engine.cycleModality();
                return;

            default:
                return;
        }

        if (action) {
            e.preventDefault();
            this.engine.step_(action);
        }
    }

    _updateSlider() {
        const slider = document.getElementById('step-size-slider');
        const label = document.getElementById('step-size-value');
        slider.value = this.moveStep;
        label.textContent = this.moveStep;
    }

    destroy() {
        document.removeEventListener('keydown', this._boundKeyDown);
        const slider = document.getElementById('step-size-slider');
        if (slider && this._sliderHandler) {
            slider.removeEventListener('input', this._sliderHandler);
        }
    }
}

// ===================================================================
// Results Modal
// ===================================================================

function showResultsModal(obs, traj, durationSec) {
    const modal = document.getElementById('results-modal');
    const result = obs.result || {};
    const ds = obs.drone_state || {};

    // Icon & title
    const iconEl = document.getElementById('results-icon');
    const titleEl = document.getElementById('results-title');
    if (result.success) {
        iconEl.textContent = '\u2714'; // checkmark
        iconEl.style.color = 'var(--success)';
        titleEl.textContent = 'Target Found!';
    } else {
        iconEl.textContent = '\u2718'; // X mark
        iconEl.style.color = 'var(--fail)';
        titleEl.textContent = 'Mission Failed';
    }

    document.getElementById('results-reason').textContent = result.reason || '';
    document.getElementById('stat-steps').textContent = traj.length;
    document.getElementById('stat-distance').textContent = Math.round(ds.total_distance || 0);
    document.getElementById('stat-duration').textContent = durationSec.toFixed(1);

    // Results minimap
    const rmCanvas = document.getElementById('results-minimap-canvas');
    if (gameEngine) {
        const rmRenderer = new MinimapRenderer(rmCanvas, gameEngine.config.map_bounds);
        traj.forEach(p => rmRenderer.addPoint(p.x, p.y));
        const meta = obs.metadata || {};
        // Show target on results minimap only for AI mode
        if (gameEngine && gameEngine.mode !== 'human' && meta.target_x != null && meta.target_y != null) {
            rmRenderer.setTarget(Number(meta.target_x), Number(meta.target_y));
        }
        const last = traj[traj.length - 1] || { x: 0, y: 0 };
        rmRenderer.render(last.x, last.y);
    }

    // Pre-fill player name from session or localStorage
    const nameInput = document.getElementById('result-player-name');
    if (session) {
        if (session.mode === 'ai' && session._modelInfo) {
            nameInput.value = session._modelInfo.model_name || '';
        } else if (session.player_name) {
            nameInput.value = session.player_name;
        } else {
            nameInput.value = localStorage.getItem('flairsim-player-name') || '';
        }
    }

    // Reset submit button
    const submitBtn = document.getElementById('btn-submit-run');
    submitBtn.disabled = false;
    submitBtn.textContent = 'Submit to Leaderboard';
    submitBtn.style.background = '';

    modal.hidden = false;
}

async function submitRun() {
    const btn = document.getElementById('btn-submit-run');
    btn.disabled = true;
    btn.textContent = 'Submitting...';

    const result = lastObservation ? lastObservation.result || {} : {};
    const ds = lastObservation ? lastObservation.drone_state || {} : {};
    const durationSec = startTime ? ((Date.now() - startTime) / 1000) : 0;
    const playerName = document.getElementById('result-player-name').value.trim();

    const metrics = computeRunMetrics(trajectory);
    const runData = {
        session_id: session ? session.session_id : null,
        scenario_id: session ? session.scenario_id : '',
        player_name: playerName || null,
        model_name: session && session.mode === 'ai' && session._modelInfo
            ? session._modelInfo.model_name : null,
        mode: session ? session.mode : 'human',
        success: result.success || false,
        reason: result.reason || '',
        steps_taken: trajectory.length,
        distance_travelled: ds.total_distance || 0,
        duration_s: durationSec,
        trajectory: trajectory,
        steps_detail: stepsDetail.length > 0 ? stepsDetail : null,
        model_info: session && session._modelInfo ? session._modelInfo : null,
        confidence: 1.0,                          // humans are always confident
        discovery_coverage: metrics.discovery_coverage,
        target_seen: result.target_in_view != null ? result.target_in_view : null,
        metrics: { discovery_coverage: metrics.discovery_coverage, unique_cells: metrics.unique_cells },
    };

    try {
        const resp = await submitToLeaderboard(runData);
        btn.textContent = 'Submitted!';
        btn.style.background = 'var(--success)';
    } catch (err) {
        btn.textContent = 'Error: ' + err.message;
        btn.disabled = false;
    }
}

/**
 * Compute discovery coverage from the trajectory array.
 * Each trajectory point: {x, y, z, step, footprint}
 *
 * Returns: {discovery_coverage, unique_cells, total_observations}
 * - discovery_coverage: unique cells / total cell observations (penalises revisits)
 */
function computeRunMetrics(traj) {
    if (!traj || traj.length === 0) {
        return { discovery_coverage: 0.0, unique_cells: 0, total_observations: 0 };
    }

    const CELL_SIZE = 10;  // 10m grid cells
    const seen = new Set();
    let totalObs = 0;

    for (const p of traj) {
        const half = (p.footprint || 0) / 2;
        if (half <= 0) continue;

        const xMin = Math.floor((p.x - half) / CELL_SIZE);
        const xMax = Math.floor((p.x + half) / CELL_SIZE);
        const yMin = Math.floor((p.y - half) / CELL_SIZE);
        const yMax = Math.floor((p.y + half) / CELL_SIZE);

        for (let cx = xMin; cx <= xMax; cx++) {
            for (let cy = yMin; cy <= yMax; cy++) {
                seen.add(cx + ',' + cy);
                totalObs++;
            }
        }
    }

    if (totalObs === 0) return { discovery_coverage: 0.0, unique_cells: 0, total_observations: 0 };

    return {
        discovery_coverage: seen.size / totalObs,
        unique_cells: seen.size,
        total_observations: totalObs,
    };
}

async function playAgain() {
    document.getElementById('results-modal').hidden = true;
    if (gameEngine) {
        await gameEngine.resetEpisode();
    }
}

// ===================================================================
// Run Detail View
// ===================================================================

async function showRunDetail(runId) {
    currentRunDetailId = runId;  // Set context before navigation
    navigateTo('run-detail');

    const titleEl = document.getElementById('detail-title');
    const metaEl = document.getElementById('detail-meta');
    const overviewEl = document.getElementById('overview-container');
    const stepsListEl = document.getElementById('detail-steps-list');
    const modelCardEl = document.getElementById('detail-model-info');
    const modelBodyEl = document.getElementById('detail-model-info-body');

    titleEl.textContent = 'Loading...';
    metaEl.innerHTML = '';
    overviewEl.innerHTML = '<div class="loading-msg">Loading overview...</div>';
    stepsListEl.innerHTML = '<div class="steps-empty">Loading...</div>';
    modelCardEl.hidden = true;

    try {
        const run = await fetchLeaderboardRun(runId);

        // Title
        const name = run.player_name || run.model_name || 'Anonymous';
        titleEl.textContent = name + ' - ' + (run.scenario_id || '');

        // Meta
        const resultClass = run.success ? 'result-success' : 'result-fail';
        const resultText = run.success ? 'SUCCESS' : 'FAIL';
        const date = run.created_at ? new Date(run.created_at).toLocaleDateString() : '';
        const distance = run.distance_travelled != null ? Math.round(run.distance_travelled) + 'm' : '-';
        const duration = run.duration_s != null ? run.duration_s.toFixed(1) + 's' : '-';
        const steps = run.steps_taken != null ? run.steps_taken + ' steps' : '-';

        metaEl.innerHTML = `
            <span class="${resultClass}">${resultText}</span>
            <span class="tag">${escapeHtml(run.mode || '')}</span>
            <span>${escapeHtml(steps)}</span>
            <span>${escapeHtml(distance)}</span>
            <span>${escapeHtml(duration)}</span>
            <span style="color:var(--text-muted)">${escapeHtml(date)}</span>
        `;
        if (run.reason) {
            metaEl.innerHTML += `<span style="color:var(--text-secondary);font-style:italic">${escapeHtml(run.reason)}</span>`;
        }

        // Overview image + trajectory SVG overlay
        await renderOverviewWithTrajectory(run, overviewEl);

        // Steps list
        renderStepsList(run, stepsListEl);

        // Model info card
        if (run.model_info && typeof run.model_info === 'object') {
            modelCardEl.hidden = false;
            let infoHtml = '';
            for (const [key, val] of Object.entries(run.model_info)) {
                infoHtml += `
                    <div class="model-info-row">
                        <span class="model-info-key">${escapeHtml(key)}</span>
                        <span class="model-info-value">${escapeHtml(String(val))}</span>
                    </div>
                `;
            }
            modelBodyEl.innerHTML = infoHtml;
        } else {
            modelCardEl.hidden = true;
        }

    } catch (err) {
        titleEl.textContent = 'Error loading run';
        overviewEl.innerHTML = '<div class="empty-msg">' + escapeHtml(err.message) + '</div>';
    }
}

async function renderOverviewWithTrajectory(run, container) {
    const scenarioId = run.scenario_id;
    if (!scenarioId) {
        container.innerHTML = '<div class="empty-msg">No scenario ID</div>';
        return;
    }

    // Fetch overview image with bounds from headers
    const overviewUrl = API_BASE + '/scenarios/' + encodeURIComponent(scenarioId) + '/overview';

    try {
        const resp = await fetch(overviewUrl);
        if (!resp.ok) {
            container.innerHTML = '<div class="empty-msg">Overview not available</div>';
            return;
        }

        const bounds = {
            x_min: parseFloat(resp.headers.get('X-Bounds-Xmin') || '0'),
            y_min: parseFloat(resp.headers.get('X-Bounds-Ymin') || '0'),
            x_max: parseFloat(resp.headers.get('X-Bounds-Xmax') || '0'),
            y_max: parseFloat(resp.headers.get('X-Bounds-Ymax') || '0'),
        };

        const blob = await resp.blob();
        const imgUrl = URL.createObjectURL(blob);

        const img = new Image();
        img.onload = () => {
            const imgW = img.naturalWidth;
            const imgH = img.naturalHeight;

            container.innerHTML = '';
            container.appendChild(img);

            // Create SVG overlay
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('viewBox', '0 0 ' + imgW + ' ' + imgH);
            svg.setAttribute('preserveAspectRatio', 'none');
            container.appendChild(svg);

            // Trajectory data
            const traj = run.trajectory || [];
            if (traj.length === 0) return;

            // World → pixel conversion
            const bw = bounds.x_max - bounds.x_min;
            const bh = bounds.y_max - bounds.y_min;

            function worldToPixel(wx, wy) {
                const px = bw > 0 ? ((wx - bounds.x_min) / bw) * imgW : imgW / 2;
                // Flip Y: world y_max = top of image (pixel y=0)
                const py = bh > 0 ? ((bounds.y_max - wy) / bh) * imgH : imgH / 2;
                return { x: px, y: py };
            }

            // Draw polyline
            const points = traj.map(p => {
                const pp = worldToPixel(p.x, p.y);
                return pp.x.toFixed(1) + ',' + pp.y.toFixed(1);
            });

            const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
            polyline.setAttribute('points', points.join(' '));
            svg.appendChild(polyline);

            // Start marker
            const startPt = worldToPixel(traj[0].x, traj[0].y);
            const startCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            startCircle.setAttribute('class', 'start-marker');
            startCircle.setAttribute('cx', startPt.x.toFixed(1));
            startCircle.setAttribute('cy', startPt.y.toFixed(1));
            startCircle.setAttribute('r', '5');
            svg.appendChild(startCircle);

            // End marker
            const lastPt = traj[traj.length - 1];
            const endPixel = worldToPixel(lastPt.x, lastPt.y);
            const endCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            endCircle.setAttribute('class', 'end-marker');
            endCircle.setAttribute('cx', endPixel.x.toFixed(1));
            endCircle.setAttribute('cy', endPixel.y.toFixed(1));
            endCircle.setAttribute('r', '5');
            svg.appendChild(endCircle);
        };
        img.src = imgUrl;

    } catch (err) {
        container.innerHTML = '<div class="empty-msg">Error loading overview: ' + escapeHtml(err.message) + '</div>';
    }
}

function renderStepsList(run, container) {
    const steps = run.steps_detail || [];
    const traj = run.trajectory || [];

    if (steps.length === 0 && traj.length === 0) {
        container.innerHTML = '<div class="steps-empty">No step data available</div>';
        return;
    }

    let html = '';

    // If we have steps_detail, use it (richer data with reasons)
    if (steps.length > 0) {
        steps.forEach((s, i) => {
            const actionType = s.action_type || 'move';
            let extraCls = '';
            if (actionType === 'found') extraCls = ' step-found';
            else if (actionType === 'stop') extraCls = ' step-stop';

            const dx = s.dx != null ? Math.round(s.dx) : 0;
            const dy = s.dy != null ? Math.round(s.dy) : 0;
            const dz = s.dz != null ? Math.round(s.dz) : 0;

            let actionText = actionType.toUpperCase();
            if (actionType === 'move') {
                actionText = 'dx=' + dx + ' dy=' + dy;
                if (dz !== 0) actionText += ' dz=' + dz;
            }

            let reasonHtml = '';
            if (s.reason) {
                reasonHtml = '<div class="step-reason">' + escapeHtml(s.reason) + '</div>';
            }

            html += `
                <div class="step-item${extraCls}">
                    <span class="step-num">${i + 1}</span>
                    <div class="step-body">
                        <div class="step-action">${escapeHtml(actionText)}</div>
                        ${reasonHtml}
                    </div>
                </div>
            `;
        });
    } else {
        // Fallback: show trajectory positions
        traj.forEach((p, i) => {
            html += `
                <div class="step-item">
                    <span class="step-num">${i + 1}</span>
                    <div class="step-body">
                        <div class="step-action">pos=(${Math.round(p.x)}, ${Math.round(p.y)}) alt=${Math.round(p.z)}m</div>
                    </div>
                </div>
            `;
        });
    }

    container.innerHTML = html;
}

// ===================================================================
// Utilities
// ===================================================================

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeAttr(str) {
    return String(str).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
}

/**
 * Return a medal emoji for rank 0-2, or the 1-indexed number otherwise.
 */
function rankBadge(index) {
    if (index === 0) return '\uD83E\uDD47'; // 🥇
    if (index === 1) return '\uD83E\uDD48'; // 🥈
    if (index === 2) return '\uD83E\uDD49'; // 🥉
    return String(index + 1);
}

// ===================================================================
// Agent Page (#agent/{name})
// ===================================================================

let _agentPageData = {}; // { scenarioId: { runs, scenarioName } }
let _agentCurrentScenario = null;
let _agentSort = null; // 3-state sort like results

async function loadAgentPage(agentName) {
    const container = document.getElementById('agent-page-content');
    container.innerHTML = '<div class="loading-msg">Loading agent data...</div>';

    try {
        if (scenarios.length === 0) {
            try { scenarios = await fetchScenarios(); } catch (_) {}
        }

        const runsData = await apiFetch('/agents/' + encodeURIComponent(agentName) + '/runs');
        const allRuns = runsData.runs || [];

        // Group runs by scenario
        _agentPageData = {};
        allRuns.forEach(r => {
            const sid = r.scenario_id;
            if (!_agentPageData[sid]) {
                const sc = scenarios.find(s => (s.scenario_id || s.id) === sid);
                _agentPageData[sid] = { runs: [], scenarioName: sc ? (sc.name || sid) : sid };
            }
            _agentPageData[sid].runs.push(r);
        });

        // Rank runs within each scenario using rank-vector method
        for (const [sid, data] of Object.entries(_agentPageData)) {
            _rankAgentRuns(data.runs);
        }

        // Header + stats
        const totalRuns = allRuns.length;
        const successRuns = allRuns.filter(r => r.success).length;
        const successRate = totalRuns > 0 ? Math.round(100 * successRuns / totalRuns) : 0;
        const scenarioCount = Object.keys(_agentPageData).length;

        let html = '<div class="agent-header"><h2>' + escapeHtml(agentName) + '</h2></div>';
        html += '<div class="agent-stats">';
        html += '<div class="stat"><span class="stat-value">' + scenarioCount + '</span><span class="stat-label">Scenarios</span></div>';
        html += '<div class="stat"><span class="stat-value">' + totalRuns + '</span><span class="stat-label">Total Runs</span></div>';
        html += '<div class="stat"><span class="stat-value">' + successRate + '%</span><span class="stat-label">Success Rate</span></div>';
        html += '</div>';

        // Scenario selector dropdown
        const sids = Object.keys(_agentPageData);
        html += '<div class="leaderboard-filters"><label>Scenario <select id="agent-scenario-select">';
        sids.forEach(sid => {
            html += '<option value="' + escapeAttr(sid) + '">' + escapeHtml(_agentPageData[sid].scenarioName) + ' (' + _agentPageData[sid].runs.length + ' runs)</option>';
        });
        html += '</select></label></div>';

        // Content area for selected scenario
        html += '<div id="agent-scenario-content"></div>';

        container.innerHTML = html;

        // Wire up selector
        const select = document.getElementById('agent-scenario-select');
        select.onchange = () => { _agentSort = null; _renderAgentScenario(select.value); };
        if (sids.length > 0) {
            _agentCurrentScenario = sids[0];
            _renderAgentScenario(sids[0]);
        }

    } catch (err) {
        container.innerHTML = '<div class="empty-msg">Error: ' + escapeHtml(err.message) + '</div>';
    }
}

function _rankAgentRuns(runs) {
    if (runs.length === 0) return;
    const lowerBetter = ['steps_taken', 'duration_s', 'distance_travelled'];
    const higherBetter = [];
    const allMetrics = lowerBetter.concat(higherBetter);
    const n = runs.length;

    // Separate success/fail
    const ok = runs.filter(r => r.success);
    const fail = runs.filter(r => !r.success);

    function assignRanks(group) {
        if (group.length === 0) return;
        const rankVecs = group.map(() => []);
        for (const metric of allMetrics) {
            const reverse = higherBetter.includes(metric);
            const vals = group.map((r, i) => [r[metric] != null ? r[metric] : (reverse ? -Infinity : Infinity), i]);
            vals.sort((a, b) => reverse ? b[0] - a[0] : a[0] - b[0]);
            const ranks = new Array(group.length);
            let curRank = 1;
            for (let pos = 0; pos < vals.length; pos++) {
                if (pos > 0 && vals[pos][0] !== vals[pos - 1][0]) curRank = pos + 1;
                ranks[vals[pos][1]] = curRank;
            }
            for (let i = 0; i < group.length; i++) {
                rankVecs[i].push(ranks[i]);
            }
        }
        // Sort vectors and attach
        group.forEach((r, i) => { r._rankVec = rankVecs[i].slice().sort((a, b) => a - b); });
        group.sort((a, b) => {
            for (let j = 0; j < a._rankVec.length; j++) {
                if (a._rankVec[j] !== b._rankVec[j]) return a._rankVec[j] - b._rankVec[j];
            }
            return 0;
        });
    }

    assignRanks(ok);
    assignRanks(fail);

    // Assign final run_rank
    const merged = ok.concat(fail);
    merged.forEach((r, i) => { r.run_rank = i + 1; delete r._rankVec; });

    // Reorder original array in-place
    runs.length = 0;
    merged.forEach(r => runs.push(r));
}

function _renderAgentScenario(scenarioId) {
    _agentCurrentScenario = scenarioId;
    const data = _agentPageData[scenarioId];
    if (!data) return;
    const runs = data.runs;
    const contentEl = document.getElementById('agent-scenario-content');

    const sortState = _agentSort;
    const sortKey = sortState ? sortState.key : 'run_rank';
    const sortAsc = sortState ? sortState.asc : true;

    const sorted = [...runs].sort((a, b) => {
        let va = a[sortKey], vb = b[sortKey];
        if (va == null) va = sortAsc ? Infinity : -Infinity;
        if (vb == null) vb = sortAsc ? Infinity : -Infinity;
        if (typeof va === 'boolean') { va = va ? 1 : 0; vb = vb ? 1 : 0; }
        return sortAsc ? va - vb : vb - va;
    });

    // Compute normalized scores across sorted runs
    _addNormalizedScores(sorted, ['steps_taken', 'duration_s', 'distance_travelled']);

    // Best values (for raw columns)
    const bestVals = {};
    const metricCols = [
        { key: 'steps_taken', best: 'min' }, { key: 'duration_s', best: 'min' },
        { key: 'distance_travelled', best: 'min' },
        { key: 'steps_taken_score', best: 'max' }, { key: 'duration_s_score', best: 'max' },
        { key: 'distance_travelled_score', best: 'max' },
    ];
    metricCols.forEach(col => {
        const vals = runs.map(r => r[col.key]).filter(v => v != null);
        if (vals.length) bestVals[col.key] = col.best === 'max' ? Math.max(...vals) : Math.min(...vals);
    });

    // Checked runs (default top 5)
    const prevChecked = _getCheckedAgents('agent');
    const checkedSet = prevChecked.length > 0
        ? new Set(prevChecked)
        : new Set(sorted.slice(0, 5).map((r, i) => 'run_' + i));

    const columns = ['run_rank', 'run_num', 'success', 'steps_taken', 'steps_taken_score', 'duration_s', 'duration_s_score', 'distance_travelled', 'distance_travelled_score', 'target_seen', 'created_at', '_check'];
    const colLabels = { '_check': 'Plot', 'run_rank': 'Rank', 'run_num': 'Run #', 'success': 'Result', 'steps_taken': 'Steps', 'steps_taken_score': 'Steps Score', 'duration_s': 'Time (s)', 'duration_s_score': 'Time Score', 'distance_travelled': 'Distance (m)', 'distance_travelled_score': 'Dist Score', 'target_seen': 'Target Seen', 'created_at': 'Date' };
    const sortable = new Set(['run_rank', 'steps_taken', 'steps_taken_score', 'duration_s', 'duration_s_score', 'distance_travelled', 'distance_travelled_score', 'target_seen', 'success']);
    const bestMap = { 'steps_taken': 'min', 'duration_s': 'min', 'distance_travelled': 'min', 'steps_taken_score': 'max', 'duration_s_score': 'max', 'distance_travelled_score': 'max' };

    let html = '<table class="leaderboard-table sortable-table"><thead><tr>';
    columns.forEach(col => {
        let arrow = '';
        if (sortState && sortState.key === col) arrow = sortState.asc ? ' ▲' : ' ▼';
        if (col === '_check') {
            html += '<th class="col-check">Plot</th>';
        } else if (sortable.has(col)) {
            html += '<th class="sortable-th" data-sort-key="' + col + '" data-scope="agent">' + colLabels[col] + arrow + '</th>';
        } else {
            html += '<th>' + colLabels[col] + '</th>';
        }
    });
    html += '</tr></thead><tbody>';

    sorted.forEach((r, idx) => {
        const runKey = 'run_' + idx;
        const color = AGENT_COLORS[idx % AGENT_COLORS.length];
        const checked = checkedSet.has(runKey) ? 'checked' : '';

        html += '<tr>';
        columns.forEach(col => {
            if (col === '_check') {
                html += '<td class="col-check"><input type="checkbox" class="agent-plot-check" data-scope="agent" data-agent="' + runKey + '" ' + checked + ' onchange="_updateAgentPlots()">'
                    + '<span class="agent-color-dot" style="background:' + color + '"></span></td>';
            } else if (col === 'run_num') {
                html += '<td>' + (runs.indexOf(r) + 1) + '</td>';
            } else if (col === 'success') {
                const cls = r.success ? 'result-success' : 'result-fail';
                html += '<td><span class="' + cls + '">' + (r.success ? 'SUCCESS' : 'FAIL') + '</span></td>';
            } else if (col === 'created_at') {
                html += '<td>' + (r.created_at ? new Date(r.created_at).toLocaleDateString() : '-') + '</td>';
            } else if (col === 'target_seen') {
                html += '<td>' + (r.target_seen ? 'Yes' : 'No') + '</td>';
            } else if (col === 'duration_s') {
                const v = r[col];
                const isBest = bestVals[col] != null && v === bestVals[col];
                html += '<td' + (isBest ? ' class="best-val"' : '') + '>' + (v != null ? v.toFixed(1) : '-') + '</td>';
            } else if (col === 'distance_travelled') {
                const v = r[col];
                const isBest = bestVals[col] != null && v === bestVals[col];
                html += '<td' + (isBest ? ' class="best-val"' : '') + '>' + (v != null ? Math.round(v) : '-') + '</td>';
            } else if (col === 'steps_taken') {
                const v = r[col];
                const isBest = bestVals[col] != null && v === bestVals[col];
                html += '<td' + (isBest ? ' class="best-val"' : '') + '>' + (v != null ? v : '-') + '</td>';
            } else if (col.endsWith('_score')) {
                const v = r[col];
                const isBest = bestVals[col] != null && v === bestVals[col];
                html += '<td' + (isBest ? ' class="best-val"' : '') + '>' + (v != null ? v.toFixed(2) : '-') + '</td>';
            } else {
                html += '<td>' + (r[col] != null ? r[col] : '-') + '</td>';
            }
        });
        html += '</tr>';
    });
    html += '</tbody></table>';

    // Plots
    html += '<div class="plots-row">';
    html += '<div class="plot-container" id="agent-parallel-' + escapeAttr(scenarioId) + '"></div>';
    html += '<div class="plot-container" id="agent-radar-' + escapeAttr(scenarioId) + '"></div>';
    html += '</div>';

    // Trajectory maps
    html += '<h3 style="margin-top:24px">Trajectories</h3>';
    html += '<div class="agent-trajectories">';
    runs.forEach((r, i) => {
        html += '<div class="agent-traj-container" id="agent-traj-' + escapeAttr(r.id) + '">';
        html += '<div class="agent-traj-label">Run ' + (i + 1) + ' - ' + (r.success ? 'Success' : 'Fail') + '</div>';
        html += '<div class="agent-traj-map"></div>';
        html += '</div>';
    });
    html += '</div>';

    contentEl.innerHTML = html;

    // Sort handlers
    contentEl.querySelectorAll('.sortable-th[data-scope="agent"]').forEach(th => {
        th.addEventListener('click', () => {
            const key = th.dataset.sortKey;
            if (!_agentSort || _agentSort.key !== key) {
                _agentSort = { key, asc: true };
            } else if (_agentSort.asc) {
                _agentSort = { key, asc: false };
            } else {
                _agentSort = null;
            }
            _renderAgentScenario(scenarioId);
        });
    });

    // Render plots
    _updateAgentPlots();

    // Render trajectory maps
    for (const r of runs) {
        const trajContainer = document.getElementById('agent-traj-' + r.id);
        if (trajContainer && r.trajectory && r.trajectory.length > 0) {
            const mapDiv = trajContainer.querySelector('.agent-traj-map');
            renderOverviewWithTrajectory(r, mapDiv);
        }
    }
}

function _updateAgentPlots() {
    const sid = _agentCurrentScenario;
    if (!sid || !_agentPageData[sid]) return;
    const runs = _agentPageData[sid].runs;

    const parallelEl = document.getElementById('agent-parallel-' + sid);
    const radarEl = document.getElementById('agent-radar-' + sid);

    const checkedKeys = _getCheckedAgents('agent');
    if (checkedKeys.length === 0) {
        if (parallelEl) parallelEl.innerHTML = '<div class="empty-msg">Select runs to compare</div>';
        if (radarEl) radarEl.innerHTML = '<div class="empty-msg">Select runs to compare</div>';
        return;
    }

    // Map checked keys (run_0, run_1, ...) to sorted indices
    const selectedRuns = checkedKeys.map(k => {
        const idx = parseInt(k.replace('run_', ''));
        // Get run from current sorted view (but we use original runs for data)
        return runs[idx] || null;
    }).filter(Boolean);

    const mapped = selectedRuns.map((r, i) => ({
        agent_name: 'Run ' + (runs.indexOf(r) + 1),
        success: r.success,
        success_rate: r.success ? 1 : 0,
        steps_taken: r.steps_taken,
        duration_s: r.duration_s,
        distance_travelled: r.distance_travelled,
        steps_score: r.steps_taken_score,
        time_score: r.duration_s_score,
        dist_score: r.distance_travelled_score,
        _colorIdx: i,
    }));

    if (parallelEl && typeof Plotly !== 'undefined') renderParallelPlot(parallelEl, mapped);
    if (radarEl && typeof Plotly !== 'undefined') renderRadarChart(radarEl, mapped);
}

// ===================================================================
// About Page
// ===================================================================

async function loadAboutPage() {
    const container = document.getElementById('about-page-content');

    if (scenarios.length === 0) {
        try { scenarios = await fetchScenarios(); } catch (_) {}
    }

    let html = '';

    html += '<h2>FlairSim Benchmark</h2>';
    html += '<p class="about-intro">';
    html += 'FlairSim is an open benchmark for evaluating autonomous AI agents on visual drone navigation tasks. ';
    html += 'Agents are dropped at a starting position over real French aerial imagery from the ';
    html += '<a href="https://ignf.github.io/FLAIR/" target="_blank" style="color:var(--accent)">FLAIR-HUB</a> ';
    html += 'dataset and must locate a specified target using only egocentric visual observations.';
    html += '</p>';

    html += '<h3>How It Works</h3>';
    html += '<p class="about-intro">';
    html += 'Each scenario defines a target object (e.g. "find the red car"), a region of interest (ROI) from real aerial imagery, ';
    html += 'and constraints like maximum steps and available sensor modalities (RGB, infrared, elevation). ';
    html += 'The agent can move in any direction, change altitude, and declare the target found when it believes it is within range.';
    html += '</p>';

    html += '<h3>Metrics</h3>';
    html += '<p class="about-intro">All runs are evaluated with flat, transparent metrics &mdash; no composite scores or hidden formulas:</p>';
    html += '<div class="about-scoring">';
    html += '<table class="leaderboard-table"><thead><tr><th>Metric</th><th>Type</th><th>Description</th></tr></thead><tbody>';
    html += '<tr><td><strong>Success</strong></td><td>Boolean</td><td>Did the agent correctly identify and declare the target?</td></tr>';
    html += '<tr><td><strong>Steps</strong></td><td>Integer</td><td>Total number of actions taken</td></tr>';
    html += '<tr><td><strong>Time</strong></td><td>Float (s)</td><td>Wall-clock duration of the episode</td></tr>';
    html += '<tr><td><strong>Distance</strong></td><td>Float (m)</td><td>Total horizontal distance travelled</td></tr>';
    html += '<tr><td><strong>Target Seen</strong></td><td>Boolean</td><td>Was the target ever within the drone\'s field of view?</td></tr>';
    html += '</tbody></table>';
    html += '</div>';

    html += '<h3>Normalized Scores</h3>';
    html += '<p class="about-intro">';
    html += 'For each numeric metric (Steps, Time, Distance), a <strong>normalized score</strong> is computed using min-max inversion: ';
    html += '<code>Score = 1 &minus; (value &minus; min) / (max &minus; min)</code>. ';
    html += 'The best agent gets 1.0, the worst gets 0.0. These scores are displayed alongside raw metrics in the tables and are used for all plots (parallel coordinates and radar charts). ';
    html += 'Higher is always better for scores.';
    html += '</p>';

    html += '<h3>Best Run Selection: Pareto Front</h3>';
    html += '<p class="about-intro">';
    html += 'Since there is no single composite score, we use <strong>Pareto optimality</strong> to select the best run per agent. ';
    html += 'A run is "Pareto-optimal" if no other run is strictly better on all criteria simultaneously.</p>';
    html += '<div class="about-scoring">';
    html += '<ol class="about-list">';
    html += '<li><strong>Filter</strong>: only successful runs are considered. If an agent has no successful runs, we pick the run with the fewest steps.</li>';
    html += '<li><strong>Pareto front</strong>: compute the set of non-dominated runs on three objectives (all minimised): steps, time, distance.</li>';
    html += '<li><strong>Compromise</strong>: from the Pareto front, select the run closest to the origin (normalised Euclidean distance). This represents the best balanced trade-off.</li>';
    html += '</ol>';
    html += '</div>';

    html += '<h3>Ranking Method: Rank-Vector Classification</h3>';
    html += '<p class="about-intro">';
    html += 'Since there is no composite score, we need a fair way to rank agents across multiple independent metrics. ';
    html += 'We use <strong>rank-vector lexicographic comparison</strong>, which works as follows:';
    html += '</p>';
    html += '<div class="about-scoring">';
    html += '<ol class="about-list">';
    html += '<li><strong>Per-metric ranking</strong>: for each metric (steps, time, distance), all agents are ranked 1 to N (lower is better for all three).</li>';
    html += '<li><strong>Rank vector</strong>: each agent collects its per-metric ranks into a vector, which is then sorted in ascending order. For example, an agent ranked 1st in steps, 3rd in time, 1st in distance gets the sorted vector <code>[1, 1, 3]</code>.</li>';
    html += '<li><strong>Lexicographic comparison</strong>: agents are compared by their sorted rank vectors element by element. <code>[1, 1, 3]</code> beats <code>[1, 2, 3]</code> because the second element (1 vs 2) breaks the tie.</li>';
    html += '<li><strong>Success first</strong>: agents with successful runs always rank above agents that failed, regardless of other metrics.</li>';
    html += '</ol>';
    html += '</div>';
    html += '<p class="about-intro">';
    html += 'This method rewards agents that are consistently good across all metrics, rather than excelling at one while being poor at others. ';
    html += 'It is transparent, requires no arbitrary weights, and produces intuitive rankings.';
    html += '</p>';

    html += '<h3>Global Results</h3>';
    html += '<p class="about-intro">';
    html += 'The global results page shows only agents that have completed <strong>every scenario</strong> in the benchmark. ';
    html += 'Each metric is averaged across all scenarios to produce a single comparable profile per agent. ';
    html += 'Use the parallel coordinates plot and radar chart to visually compare agents.';
    html += '</p>';

    html += '<h3>Using FlairSim for AI Agents</h3>';
    html += '<ol class="about-list">';
    html += '<li><strong>Create a session</strong>: <code>POST /api/sessions</code> with scenario_id, mode "ai", and model info.</li>';
    html += '<li><strong>Reset</strong>: <code>POST /api/sessions/{id}/sim/reset</code> to start the episode.</li>';
    html += '<li><strong>Step</strong>: <code>POST /api/sessions/{id}/sim/step</code> with dx, dy, dz displacements or action_type "found".</li>';
    html += '<li><strong>Submit</strong>: <code>POST /api/leaderboard/submit</code> with your run results.</li>';
    html += '</ol>';

    // Scenarios table
    if (scenarios.length > 0) {
        html += '<h3>Available Scenarios</h3>';
        html += '<div class="leaderboard-table-container"><table class="leaderboard-table"><thead><tr>';
        html += '<th>Name</th><th>Domain</th><th>Environment</th><th>Difficulty</th><th>Max Steps</th>';
        html += '</tr></thead><tbody>';
        scenarios.forEach(s => {
            const name = s.name || s.scenario_id || s.id;
            const domain = s.dataset ? (s.dataset.domain || '-') : '-';
            const env = s.environment ? s.environment.join(', ') : '-';
            const diff = s.difficulty || 0;
            let stars = '';
            for (let i = 1; i <= 3; i++) stars += i <= diff ? '\u2605' : '\u2606';
            html += '<tr>';
            html += '<td>' + escapeHtml(name) + '</td>';
            html += '<td>' + escapeHtml(domain) + '</td>';
            html += '<td>' + escapeHtml(env) + '</td>';
            html += '<td>' + stars + '</td>';
            html += '<td>' + (s.max_steps || '-') + '</td>';
            html += '</tr>';
        });
        html += '</tbody></table></div>';
    }

    html += '<h3>Links</h3>';
    html += '<div class="about-links">';
    html += '<a href="https://ignf.github.io/FLAIR/" target="_blank" class="about-link-btn">FLAIR-HUB Dataset</a>';
    html += '</div>';

    container.innerHTML = html;
}

// ===================================================================
// Init
// ===================================================================

function init() {
    // Route handling
    window.addEventListener('hashchange', handleRoute);

    // Initial route
    handleRoute();
}

document.addEventListener('DOMContentLoaded', init);
