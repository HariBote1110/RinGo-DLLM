Chart.defaults.color = '#8888a0';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.size = 11;

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
});

const chartOpts = (yMin, yMax, yPos) => ({
    responsive: true, maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { labels: { boxWidth: 10, padding: 8 } } },
    scales: yPos ? {
        y: { position: 'left', min: yMin, max: yMax },
        y1: { position: 'right', grid: { drawOnChartArea: false } }
    } : {
        y: { min: yMin, max: yMax }
    }
});

function makeChart(id, datasets, opts) {
    const ctx = document.getElementById(id).getContext('2d');
    return new Chart(ctx, { type: 'line', data: { labels: [], datasets }, options: opts });
}

const lossChart = makeChart('lossChart', [
    { label: 'Loss', data: [], borderColor: '#a78bfa', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
    { label: 'Avg Loss', data: [], borderColor: '#34d399', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
], chartOpts(undefined, undefined, false));

const tputChart = makeChart('tputChart', [
    { label: 'tok/s', data: [], borderColor: '#fb923c', backgroundColor: 'rgba(251,146,60,0.08)', fill: true, borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
], chartOpts(0, undefined, false));

const gpuChart = makeChart('gpuChart', [
    { label: 'GPU %', data: [], borderColor: '#f87171', yAxisID: 'y', borderWidth: 1.5, pointRadius: 0 },
    { label: 'VRAM App (MB, approx)', data: [], borderColor: '#22d3ee', yAxisID: 'y1', borderWidth: 1.5, pointRadius: 0 },
], chartOpts(0, 100, true));

const cpuChart = makeChart('cpuChart', [
    { label: 'CPU App %', data: [], borderColor: '#f87171', yAxisID: 'y', borderWidth: 1.5, pointRadius: 0 },
    { label: 'RAM App (MB)', data: [], borderColor: '#22d3ee', yAxisID: 'y1', borderWidth: 1.5, pointRadius: 0 },
], chartOpts(0, undefined, true));

const pcieChart = makeChart('pcieChart', [
    { label: 'TX (MB/s)', data: [], borderColor: '#fb923c', borderWidth: 1.5, pointRadius: 0 },
    { label: 'RX (MB/s)', data: [], borderColor: '#a78bfa', borderWidth: 1.5, pointRadius: 0 },
], chartOpts(0, undefined, false));

const MAX_CHART_POINTS = 600;
function trimChart(chart) {
    if (chart.data.labels.length > MAX_CHART_POINTS) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(d => d.data.shift());
    }
}

const $ = id => document.getElementById(id);
const bySelectorAll = selector => Array.from(document.querySelectorAll(selector));
const fmt = (v, d = 1) => (v != null && isFinite(v)) ? v.toFixed(d) : '-';
const fmtInt = v => (v != null) ? v.toLocaleString() : '-';
const fmtDec = (v, d = 6) => {
    if (v == null || !isFinite(v)) return '-';
    if (v === 0) return '0';
    const abs = Math.abs(v);
    if (abs < 1e-10) return v.toExponential(2);
    if (abs < 0.001) return v.toFixed(8);
    return v.toFixed(d);
};
const fmtSci = v => {
    if (v == null || !isFinite(v)) return '-';
    if (v === 0) return '0';
    return v.toExponential(4);
}

const getFilterCheckboxes = () => bySelectorAll('.log-filter-checkbox');
const getAvailableLogFilters = () => {
    const fromState = currentState?.controls?.available_log_filters;
    if (Array.isArray(fromState) && fromState.length > 0) {
        return fromState.map(v => String(v).toLowerCase());
    }
    return getFilterCheckboxes().map(input => String(input.value).toLowerCase());
};
const renderLogFilterCheckboxes = (filters) => {
    const group = $('logFilterGroup');
    if (!group) return;
    const normalized = (filters || []).map(v => String(v).toLowerCase());
    if (normalized.length === 0) return;
    group.innerHTML = normalized.map(filter => (
        `<label class="check-item"><input type="checkbox" class="log-filter-checkbox" value="${filter}" />${filter}</label>`
    )).join('');
};
const getSelectedFilters = () => getFilterCheckboxes().filter(input => input.checked).map(input => input.value);
const setSelectedFilters = (filters) => {
    const selected = new Set((filters || []).map(v => String(v).toLowerCase()));
    getFilterCheckboxes().forEach(input => {
        input.checked = selected.has(input.value.toLowerCase());
    });
};
const updateLogMeta = () => {
    const selected = getSelectedFilters();
    const knownSet = new Set(getAvailableLogFilters());
    const unknown = (currentState?.controls?.log_filters || []).filter(v => !knownSet.has(String(v).toLowerCase()));
    const list = [...selected, ...unknown];
    const lineCount = $('logsPanel').textContent.split('\n').filter(Boolean).length;
    $('logsMeta').textContent = `フィルタ: ${list.length > 0 ? list.join(', ') : 'all'} / 表示行数: ${lineCount}`;
};
const fmtVram = v => {
    if (v == null || v <= 0) return 'N/A (WDDM/driver)';
    return fmt(v, 0) + ' MB';
};
const fmtSec = s => {
    if (!s || !isFinite(s) || s <= 0) return '-';
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    if (h > 0) return `${h}h ${m}m ${sec}s`;
    if (m > 0) return `${m}m ${sec}s`;
    return `${sec}s`;
};

function setRows(tableId, rows) {
    const tbody = $(tableId).querySelector('tbody') || $(tableId);
    tbody.innerHTML = rows.map(([k, v, t]) =>
        `<tr><td>${k}</td><td class="mono">${v}</td>${t ? `<td class="mono">${t}</td>` : ''}</tr>`
    ).join('');
}

let currentState = null;
let wsClient = null;

function sendCommand(payload) {
    if (!wsClient || wsClient.readyState !== WebSocket.OPEN) return;
    wsClient.send(JSON.stringify(payload));
}

function syncControls(s) {
    if (!s.controls) return;
    renderLogFilterCheckboxes(s.controls.available_log_filters || []);
    $('logLevelSelect').value = (s.controls.log_level || 'info').toLowerCase();
    setSelectedFilters(s.controls.log_filters || []);
    $('breakpointInput').value = (s.controls.breakpoint_events || []).join(',');
    $('pauseState').textContent = s.controls.paused ? 'paused' : 'running';
    if (s.logs) {
        $('logsPanel').textContent = s.logs.join('\n');
        $('logsPanel').scrollTop = $('logsPanel').scrollHeight;
    }
    updateLogMeta();
}

function bindControls() {
    $('logLevelSelect').addEventListener('change', (e) => {
        sendCommand({ type: 'set_log_level', level: e.target.value });
    });
    $('logFilterGroup').addEventListener('change', (event) => {
        if (!event.target.classList.contains('log-filter-checkbox')) {
            return;
        }
            const filters = getSelectedFilters();
            sendCommand({ type: 'set_log_filters', filters });
            updateLogMeta();
    });
    $('breakpointInput').addEventListener('change', (e) => {
        const events = e.target.value.split(',').map(v => v.trim()).filter(Boolean);
        sendCommand({ type: 'set_breakpoints', events });
    });
    $('pauseBtn').addEventListener('click', () => {
        sendCommand({ type: 'set_paused', paused: true });
    });
    $('resumeBtn').addEventListener('click', () => {
        sendCommand({ type: 'set_paused', paused: false });
    });
}

function updateVramSemanticsText(vramProcMb) {
    const msg = (vramProcMb != null && vramProcMb > 0)
        ? 'VRAM(App) はプロセス単位の推定値です。WDDM 等ではサンプリング毎に揺れる場合があります。'
        : 'VRAM(App) は未取得です。WDDM/ドライバ制約により取得不可または不安定な場合があります。';
    $('vramSemantics').textContent = msg;
}

function updateOverviewTab(s) {
    if (s.dataset) {
        const d = s.dataset;
        $('dSamples').textContent = fmtInt(d.total_samples);
        $('dTotalToks').textContent = fmtInt(d.total_tokens);
        $('dValidToks').textContent = fmtInt(d.valid_tokens);
        $('dSeqLen').textContent = fmtInt(d.max_seq_len);
    }

    const cfg = s.config;
    const cfgRows = [
        // ── Architecture ──
        ['vocab_size',        fmtInt(cfg.vocab_size),        'int'],
        ['hidden_dim',        fmtInt(cfg.hidden_dim),        'int'],
        ['num_layers',        cfg.num_layers,                'int'],
        ['num_heads',         cfg.num_heads,                 'int'],
        ['ffn_dim',           fmtInt(cfg.ffn_dim),           'int'],
        ['max_seq_len',       cfg.max_seq_len,               'int'],
        // ── Diffusion ──
        ['T (noise steps)',   cfg.T,                         'int'],
        ['mask_schedule',     cfg.mask_schedule   || '-',    'str'],
        ['mask_loss_weight',  fmt(cfg.mask_loss_weight, 1),  'float'],
        // ── Training ──
        ['num_epochs',        cfg.num_epochs,                'int'],
        ['batch_size',        cfg.batch_size,                'int'],
        ['learning_rate',     fmtDec(cfg.learning_rate),     'float'],
        ['lr_min',            fmtDec(cfg.lr_min),            'float'],
        ['lr_schedule',       cfg.lr_schedule     || '-',    'str'],
        ['warmup_steps',      fmtInt(cfg.warmup_steps),      'int'],
        ['weight_decay',      fmtDec(cfg.weight_decay),      'float'],
        ['grad_clip',         fmt(cfg.grad_clip, 1),         'float'],
        // ── Data / Tokeniser ──
        ['dataset_name',      cfg.dataset_name    || '-',    'str'],
        ['tokenizer_name',    cfg.tokenizer_name  || '-',    'str'],
        ['checkpoint_dir',    cfg.checkpoint_dir  || '-',    'str'],
    ];
    setRows('configTable', cfgRows);

    if (s.module_resources && s.module_resources.length > 0) {
        const tbody = $('moduleTable').querySelector('tbody');
        tbody.innerHTML = s.module_resources.map(m => `
            <tr>
                <td><strong>${m.name}</strong></td>
                <td class="mono">${fmtSci(m.flops)}</td>
                <td class="mono">${fmt(m.vram_weight_mb, 2)}</td>
                <td class="mono">${fmt(m.vram_activation_mb, 2)}</td>
                <td class="mono">${fmt(m.ram_weight_mb, 2)}</td>
                <td class="mono">${fmt(m.pcie_transfer_mb, 2)}</td>
            </tr>
        `).join('');
    }
}

function updateTrainingTab(s) {
    const p = s.progress;
    $('mEpoch').textContent = p.total_epochs > 0 ? `${p.epoch} / ${p.total_epochs}` : p.epoch;
    $('mStep').textContent = p.steps_per_epoch > 0 ? `${p.step} / ${p.steps_per_epoch}` : p.step;
    $('mLoss').textContent = fmtDec(p.loss, 4);
    $('mLR').textContent = fmtDec(p.learning_rate) || '-';
    $('mETA').textContent = fmtSec(p.eta_sec);

    const pct = p.total_steps > 0 ?
        (((p.epoch > 0 ? (p.epoch - 1) : 0) * p.steps_per_epoch + p.step) / p.total_steps * 100) : 0;
    $('progressFill').style.width = Math.min(pct, 100) + '%';
    const doneSteps = (p.epoch > 0 ? (p.epoch - 1) : 0) * p.steps_per_epoch + p.step;
    $('progressText').textContent = `${fmtInt(doneSteps)} / ${fmtInt(p.total_steps)} steps (${fmt(pct)}%)`;
}

function updateHardwareTab(s) {
    const hw = s.hardware;
    const pr = s.process;
    $('hGpu').textContent = fmt(hw.gpu_utilization) + ' %';
    $('hVram').textContent = `${fmt(hw.vram_used_mb, 0)} / ${fmt(hw.vram_total_mb, 0)} MB`;
    $('hVramProc').textContent = fmtVram(pr.vram_used_mb);
    updateVramSemanticsText(pr.vram_used_mb);

    $('hCpuProc').textContent = fmt(pr.cpu_usage_percent) + ' %';
    $('hRam').textContent = `${fmt(hw.ram_used_mb, 0)} / ${fmt(hw.ram_total_mb, 0)} MB`;
    $('hRamProc').textContent = fmt(pr.ram_used_mb, 0) + ' MB';

    $('hTx').textContent = fmt(hw.pcie_tx_mb, 2) + ' MB/s';
    $('hRx').textContent = fmt(hw.pcie_rx_mb, 2) + ' MB/s';
}

function appendChartPoint(pt) {
    const label = pt.global_step > 0 ? pt.global_step : Math.floor(pt.t);
    if (pt.loss > 0) {
        trimChart(lossChart);
        lossChart.data.labels.push(label);
        lossChart.data.datasets[0].data.push(pt.loss);
        lossChart.data.datasets[1].data.push(pt.avg_loss);
        lossChart.update('none');
    }
    if (pt.tok_s > 0) {
        trimChart(tputChart);
        tputChart.data.labels.push(label);
        tputChart.data.datasets[0].data.push(pt.tok_s);
        tputChart.update('none');
    }
    trimChart(gpuChart);
    gpuChart.data.labels.push(Math.floor(pt.t));
    gpuChart.data.datasets[0].data.push(pt.gpu_sys);
    gpuChart.data.datasets[1].data.push(pt.vram_proc > 0 ? pt.vram_proc : null);
    gpuChart.update('none');

    trimChart(cpuChart);
    cpuChart.data.labels.push(Math.floor(pt.t));
    cpuChart.data.datasets[0].data.push(pt.cpu_proc);
    cpuChart.data.datasets[1].data.push(pt.ram_proc);
    cpuChart.update('none');

    trimChart(pcieChart);
    pcieChart.data.labels.push(Math.floor(pt.t));
    pcieChart.data.datasets[0].data.push(pt.pcie_tx);
    pcieChart.data.datasets[1].data.push(pt.pcie_rx);
    pcieChart.update('none');
}

function loadHistory(points) {
    const allCharts = [lossChart, tputChart, gpuChart, cpuChart, pcieChart];
    allCharts.forEach(c => {
        c.data.labels = [];
        c.data.datasets.forEach(d => { d.data = []; });
    });
    points.forEach(pt => appendChartPoint(pt));
}

function connectWs() {
    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    wsClient = ws;
    ws.onopen = () => {
        $('connText').textContent = 'Connected';
        $('connDot').classList.add('on');
    };
    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            const handlers = {
                init: (payload) => {
                    currentState = payload.state;
                    updateOverviewTab(currentState);
                    updateTrainingTab(currentState);
                    updateHardwareTab(currentState);
                    syncControls(currentState);
                },
                update: (payload) => {
                    currentState = payload.state;
                    updateOverviewTab(currentState);
                    updateTrainingTab(currentState);
                    updateHardwareTab(currentState);
                    syncControls(currentState);
                },
                history: (payload) => loadHistory(payload.points || []),
                history_append: (payload) => {
                    if (payload.point) appendChartPoint(payload.point);
                },
                logs: (payload) => {
                    if (!payload.lines) return;
                    $('logsPanel').textContent = payload.lines.join('\n');
                    $('logsPanel').scrollTop = $('logsPanel').scrollHeight;
                    updateLogMeta();
                }
            };
            handlers[msg.type]?.(msg);
        } catch (e) {
            console.error('Parse error:', e);
        }
    };
    ws.onclose = () => {
        $('connText').textContent = 'Disconnected';
        $('connDot').classList.remove('on');
        setTimeout(connectWs, 2000);
    };
}
bindControls();
connectWs();
