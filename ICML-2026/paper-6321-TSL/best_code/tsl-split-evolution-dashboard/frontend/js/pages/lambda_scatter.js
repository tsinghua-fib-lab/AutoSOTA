(async function () {
    const runSelect = document.getElementById('runSelect');
    const epochSelect = document.getElementById('epochSelect');
    const maxTreesInput = document.getElementById('maxTrees');
    const logScaleToggle = document.getElementById('logScaleToggle');
    const scatterContainer = document.getElementById('lambdaScatter');
    const histogramContainer = document.getElementById('lambdaHistogram');
    const scatterHelpers = window.LambdaScatterHelpers;

    if (!runSelect || !epochSelect || !scatterContainer || !scatterHelpers) {
        console.warn('Lambda scatter page initialized without required elements/helpers');
        return;
    }

    const { sanitizeMaxTrees, renderScatter, renderHistogramSummary } = scatterHelpers;
    const lambdaCache = ComponentPageShared.createAsyncCache();
    const refreshScheduler = ComponentPageShared.createRenderScheduler(() => refreshScatter());
    const refreshSeq = ComponentPageShared.createRenderSequence();

    const epochsByRun = {};

    function cacheKey(runId, epoch, maxTrees) {
        return `${runId}:${epoch}:${maxTrees}`;
    }

    function showMessage(message) {
        scatterContainer.innerHTML = `<p>${message}</p>`;
        histogramContainer.innerHTML = '';
    }

    async function loadRuns() {
        try {
            const runs = await UI.loadRunsTo(runSelect);
            if (runs.length > 0) {
                runSelect.value = runs[0].run_id;
                await loadEpochs();
            }
        } catch (err) {
            console.error('Failed to load runs', err);
            showMessage('Failed to load runs.');
        }
    }

    async function loadEpochs() {
        const runId = +runSelect.value;
        if (!Number.isFinite(runId)) return;

        try {
            const payload = await API.getEpochsTrees(runId);
            epochsByRun[runId] = payload;
            const epochs = Object.keys(payload || {}).map(e => Number(e)).sort((a, b) => a - b);
            if (epochs.length === 0) {
                epochSelect.innerHTML = '';
                showMessage('No epochs available for this run.');
                return;
            }
            UI.setOptions(epochSelect, epochs.map(e => ({ label: `Stage ${e}`, value: e })));
            epochSelect.value = epochs[0];
            await refreshScatter();
        } catch (err) {
            console.error('Failed to load epochs/trees', err);
            showMessage('Error loading epochs.');
        }
    }

    async function ensureLambdas(runId, epoch, maxTrees) {
        const key = cacheKey(runId, epoch, maxTrees);
        return lambdaCache.getOrCreate(key, () => API.getTensorLambdas(runId, epoch, maxTrees));
    }

    async function refreshScatter() {
        const seq = refreshSeq.next();
        const runId = +runSelect.value;
        const epoch = +epochSelect.value;
        const maxTrees = sanitizeMaxTrees(maxTreesInput.value);
        const logScale = !!logScaleToggle.checked;

        if (![runId, epoch].every(Number.isFinite)) return;
        maxTreesInput.value = maxTrees;

        showMessage('Loading…');
        try {
            const payload = await ensureLambdas(runId, epoch, maxTrees);
            if (!refreshSeq.isCurrent(seq)) return;
            const rows = Array.isArray(payload?.trees) ? payload.trees : [];
            // Also fetch combination choice (best/candidates) for this epoch if available
            let choice = null;
            try {
                choice = await API.getCombinationChoice(runId, epoch);
            } catch (e) {
                // ignore missing endpoint or data
                choice = null;
            }
            renderScatter(scatterContainer, rows, logScale, choice);
            renderHistogramSummary(histogramContainer, rows);
        } catch (err) {
            console.error('Failed to fetch lambda scatter data', err);
            if (refreshSeq.isCurrent(seq)) {
                showMessage('Failed to load lambda data.');
            }
        }
    }

    runSelect.addEventListener('change', async () => {
        await loadEpochs();
    });
    epochSelect.addEventListener('change', () => refreshScheduler.schedule());
    maxTreesInput.addEventListener('input', () => refreshScheduler.schedule(200));
    maxTreesInput.addEventListener('blur', () => refreshScheduler.schedule());
    logScaleToggle.addEventListener('change', () => refreshScheduler.schedule());

    await loadRuns();
})();
