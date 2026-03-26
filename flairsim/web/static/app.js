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
    }
    currentView = view;

    if (view === 'landing') {
        loadScenarios();
    } else if (view === 'processes') {
        loadProcesses();
    } else if (view === 'leaderboard') {
        loadGlobalLeaderboard();
    }

    window.location.hash = view;
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

async function fetchLeaderboard(scenarioId, mode, limit) {
    let qs = '?limit=' + (limit || 50);
    if (scenarioId) qs += '&scenario_id=' + encodeURIComponent(scenarioId);
    if (mode) qs += '&mode=' + encodeURIComponent(mode);
    return apiFetch('/leaderboard' + qs);
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

        // Load mini-leaderboards in parallel (fire-and-forget per card)
        scenarios.forEach(s => {
            const id = s.scenario_id || s.id;
            loadMiniLeaderboard(id);
        });
    } catch (err) {
        grid.innerHTML = '<div class="empty-msg">Error loading scenarios: ' + escapeHtml(err.message) + '</div>';
    }
}

function renderScenarioCard(s) {
    const id = s.scenario_id || s.id;
    const name = s.name || id;
    const desc = s.description || '';
    const domain = s.dataset ? s.dataset.domain : '';
    const modalities = s.dataset && s.dataset.modalities ? s.dataset.modalities : [];
    const maxSteps = s.max_steps || '?';
    const objective = s.objective || '';
    const environment = s.environment || [];
    const difficulty = s.difficulty || 0;

    // Tags
    let tags = '';
    if (domain) tags += '<span class="tag tag-domain">' + escapeHtml(domain) + '</span>';
    environment.forEach(env => {
        tags += '<span class="tag tag-env">' + escapeHtml(env) + '</span>';
    });
    modalities.forEach(m => {
        tags += '<span class="tag">' + escapeHtml(m) + '</span>';
    });
    tags += '<span class="tag">max ' + maxSteps + ' steps</span>';
    if (objective) tags += '<span class="tag">' + escapeHtml(objective) + '</span>';

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

    return `
        <div class="scenario-card">
            <h3>${escapeHtml(name)} ${starsHtml}</h3>
            <p class="scenario-desc">${escapeHtml(desc)}</p>
            <div class="scenario-meta">${tags}</div>
            <div class="mini-leaderboard" id="mini-lb-${escapeAttr(id)}">
                <h4>Top Runs</h4>
                <div class="mini-leaderboard-empty">Loading...</div>
            </div>
            <div class="scenario-actions">
                <button class="btn btn-primary" onclick="startPlay('${escapeAttr(id)}')">
                    Play
                </button>
                <button class="btn btn-ghost" onclick="showScenarioLeaderboard('${escapeAttr(id)}')">
                    Leaderboard
                </button>
            </div>
        </div>
    `;
}

async function loadMiniLeaderboard(scenarioId) {
    const container = document.getElementById('mini-lb-' + scenarioId);
    if (!container) return;

    try {
        const data = await apiFetch('/leaderboard/scenario/' + encodeURIComponent(scenarioId) + '?limit=5');
        const runs = data.runs || [];
        if (runs.length === 0) {
            container.innerHTML = '<h4>Top Runs</h4><div class="mini-leaderboard-empty">No runs yet</div>';
            return;
        }
        let html = '<h4>Top Runs</h4>';
        runs.forEach((r, i) => {
            const name = r.player_name || r.model_name || 'Anon';
            const resultCls = r.success ? 'result-success' : 'result-fail';
            const resultTxt = r.success ? 'OK' : 'FAIL';
            const score = r.score != null ? r.score.toFixed(0) : '-';
            html += `
                <div class="mini-lb-row" onclick="showRunDetail('${escapeAttr(r.id)}')" style="cursor:pointer" title="View run detail">
                    <span class="mini-lb-rank">${i + 1}.</span>
                    <span class="mini-lb-name">${escapeHtml(name)}</span>
                    <span class="mini-lb-score">${escapeHtml(score)}</span>
                    <span class="mini-lb-result ${resultCls}">${resultTxt}</span>
                </div>
            `;
        });
        container.innerHTML = html;
    } catch (_) {
        container.innerHTML = '<h4>Top Runs</h4><div class="mini-leaderboard-empty">--</div>';
    }
}

// ===================================================================
// Play Session Management
// ===================================================================

async function startPlay(scenarioId) {
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

        // Create session
        session = await createSession(scenarioId, 'human');
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

function showScenarioLeaderboard(scenarioId) {
    navigateTo('leaderboard');
    // Switch to scenario tab and set the filter to this scenario
    setTimeout(() => {
        switchLbTab('scenario');
        const select = document.getElementById('lb-filter-scenario');
        if (select) {
            select.value = scenarioId;
            loadLeaderboard();
        }
    }, 50);
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

function switchLbTab(tab) {
    const tabs = ['global', 'scenario'];
    tabs.forEach(t => {
        document.getElementById('lb-tab-' + t).classList.toggle('active', t === tab);
        document.getElementById('lb-panel-' + t).hidden = t !== tab;
    });
    if (tab === 'global') {
        loadGlobalLeaderboard();
    } else {
        loadLeaderboard();
    }
}

async function loadGlobalLeaderboard() {
    const container = document.getElementById('lb-global-container');
    container.innerHTML = '<div class="loading-msg">Loading...</div>';
    try {
        const data = await apiFetch('/leaderboard/global?limit=100');
        const entries = data.leaderboard || [];
        if (entries.length === 0) {
            container.innerHTML = '<div class="empty-msg">No runs yet.</div>';
            return;
        }
        let html = `
            <table class="leaderboard-table">
                <thead><tr>
                    <th>#</th>
                    <th>Agent</th>
                    <th>Score</th>
                    <th>Scenarios</th>
                </tr></thead>
                <tbody>
        `;
        entries.forEach((e, i) => {
            const score = e.total_score != null ? e.total_score.toFixed(1) : '-';
            html += `<tr>
                <td>${i + 1}</td>
                <td>${escapeHtml(e.agent_name || '-')}</td>
                <td>${escapeHtml(score)}</td>
                <td>${e.scenarios_attempted || 0}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        container.innerHTML = html;
    } catch (err) {
        container.innerHTML = '<div class="empty-msg">Error: ' + escapeHtml(err.message) + '</div>';
    }
}

async function loadLeaderboard() {
    const container = document.getElementById('leaderboard-table-container');
    const scenarioSelect = document.getElementById('lb-filter-scenario');
    const modeSelect = document.getElementById('lb-filter-mode');

    // Populate scenario filter options
    if (scenarios.length === 0) {
        try { scenarios = await fetchScenarios(); } catch (_) {}
    }
    // Only repopulate if empty (except the default option)
    if (scenarioSelect.options.length <= 1) {
        scenarios.forEach(s => {
            const id = s.scenario_id || s.id;
            const name = s.name || id;
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = name;
            scenarioSelect.appendChild(opt);
        });
        // Attach filter change handlers
        scenarioSelect.onchange = loadLeaderboard;
        modeSelect.onchange = loadLeaderboard;
    }

    const scenarioId = scenarioSelect.value || null;
    const mode = modeSelect.value || null;

    try {
        let runs = [];
        let usedScoredEndpoint = false;

        if (scenarioId) {
            // Use the scored per-scenario endpoint when a scenario is selected
            const data = await apiFetch('/leaderboard/scenario/' + encodeURIComponent(scenarioId) + '?limit=100');
            runs = data.runs || [];
            usedScoredEndpoint = true;
            // Apply mode filter client-side since scored endpoint doesn't support it
            if (mode) {
                runs = runs.filter(r => r.mode === mode);
            }
        } else {
            const data = await fetchLeaderboard(scenarioId, mode, 100);
            runs = data.runs || [];
        }

        if (runs.length === 0) {
            container.innerHTML = '<div class="empty-msg">No runs found.</div>';
            return;
        }

        let html = `
            <table class="leaderboard-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Player</th>
                        <th>Scenario</th>
                        <th>Mode</th>
                        <th>Result</th>
                        <th>Score</th>
                        <th>Steps</th>
                        <th>Distance</th>
                        <th>Duration</th>
                        <th>Date</th>
                    </tr>
                </thead>
                <tbody>
        `;

        runs.forEach((r, i) => {
            const name = r.player_name || r.model_name || 'Anon';
            const resultCls = r.success ? 'result-success' : 'result-fail';
            const resultTxt = r.success ? 'SUCCESS' : 'FAIL';
            const steps = r.steps_taken != null ? r.steps_taken : '-';
            const dist = r.distance_travelled != null ? Math.round(r.distance_travelled) + 'm' : '-';
            const dur = r.duration_s != null ? formatDuration(r.duration_s) : '-';
            const date = r.created_at ? new Date(r.created_at).toLocaleDateString() : '-';
            const scenarioName = (() => {
                const sc = scenarios.find(s => (s.scenario_id || s.id) === r.scenario_id);
                return sc ? (sc.name || r.scenario_id) : r.scenario_id;
            })();
            const modeLabel = r.mode === 'ai' ? 'AI' : 'Human';
            const score = r.score != null ? r.score.toFixed(1) : '-';

            html += `
                <tr class="lb-row" onclick="showRunDetail('${escapeAttr(r.id)}')" style="cursor:pointer" title="View run detail">
                    <td>${i + 1}</td>
                    <td>${escapeHtml(name)}</td>
                    <td>${escapeHtml(scenarioName)}</td>
                    <td>${modeLabel}</td>
                    <td><span class="${resultCls}">${resultTxt}</span></td>
                    <td>${escapeHtml(score)}</td>
                    <td>${steps}</td>
                    <td>${dist}</td>
                    <td>${dur}</td>
                    <td>${date}</td>
                </tr>
            `;
        });

        html += '</tbody></table>';
        container.innerHTML = html;

    } catch (err) {
        container.innerHTML = '<div class="empty-msg">Error: ' + escapeHtml(err.message) + '</div>';
    }
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

    // Pre-fill player name from session
    const nameInput = document.getElementById('result-player-name');
    if (session) {
        if (session.mode === 'ai' && session._modelInfo) {
            nameInput.value = session._modelInfo.model_name || '';
        } else if (session.player_name) {
            nameInput.value = session.player_name;
        } else {
            nameInput.value = '';
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
        fov_coverage: metrics.fov_coverage,
        target_seen: result.target_in_view != null ? result.target_in_view : null,
        metrics: { fov_coverage: metrics.fov_coverage, unique_cells: metrics.unique_cells_count },
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
 * Compute run metrics from the trajectory array.
 * Each trajectory point: {x, y, z, step, footprint}
 *
 * Returns: {fov_coverage, unique_cells_count}
 * - fov_coverage: fraction of unique 10m grid cells covered vs total cells in bounding box
 *   (0.0 if trajectory empty or bounding box has zero area)
 * - unique_cells_count: number of distinct 10m grid cells covered
 */
function computeRunMetrics(traj) {
    if (!traj || traj.length === 0) {
        return { fov_coverage: 0.0, unique_cells_count: 0 };
    }

    const CELL_SIZE = 10;  // 10m grid cells
    const coveredCells = new Set();

    for (const p of traj) {
        const halfWidth = (p.footprint || 0) / 2;
        if (halfWidth <= 0) continue;

        // Cells covered by this footprint square centered at (p.x, p.y)
        const xMin = Math.floor((p.x - halfWidth) / CELL_SIZE);
        const xMax = Math.floor((p.x + halfWidth) / CELL_SIZE);
        const yMin = Math.floor((p.y - halfWidth) / CELL_SIZE);
        const yMax = Math.floor((p.y + halfWidth) / CELL_SIZE);

        for (let cx = xMin; cx <= xMax; cx++) {
            for (let cy = yMin; cy <= yMax; cy++) {
                coveredCells.add(cx + ',' + cy);
            }
        }
    }

    // Bounding box of the trajectory
    const xs = traj.map(p => p.x);
    const ys = traj.map(p => p.y);
    const xSpan = Math.max(...xs) - Math.min(...xs);
    const ySpan = Math.max(...ys) - Math.min(...ys);

    const totalCellsX = Math.max(1, Math.ceil((xSpan + 2 * (traj[0].footprint || 0)) / CELL_SIZE));
    const totalCellsY = Math.max(1, Math.ceil((ySpan + 2 * (traj[0].footprint || 0)) / CELL_SIZE));
    const totalCells = totalCellsX * totalCellsY;

    return {
        fov_coverage: Math.min(1.0, coveredCells.size / totalCells),
        unique_cells_count: coveredCells.size,
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

        // Try to get computed score for this run
        try {
            const scoredData = await apiFetch('/leaderboard/scenario/' + encodeURIComponent(run.scenario_id) + '?limit=200');
            const scoredRun = (scoredData.runs || []).find(r => r.id === run.id);
            if (scoredRun && scoredRun.score != null) {
                const scoreEl = document.createElement('span');
                scoreEl.className = 'score-badge';
                scoreEl.textContent = 'Score: ' + scoredRun.score.toFixed(1);
                metaEl.appendChild(scoreEl);
            }
        } catch (_) {
            // Score display is optional
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
    return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
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
