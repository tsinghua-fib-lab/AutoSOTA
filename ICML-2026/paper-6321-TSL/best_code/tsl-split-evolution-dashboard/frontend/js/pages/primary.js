(async function () {
    const combinedStage = document.getElementById('combinedStage');
    const combinedRuns = document.getElementById('combinedRuns');

    if (!combinedStage || !combinedRuns) {
        console.error('Required elements not found');
        return;
    }

    async function loadEpochs() {
        try {
            const epochs = await API.getCombinedEpochs();
            UI.setOptions(combinedStage, epochs.map(e => ({ label: `Stage ${e}`, value: e })));
            if (epochs.length > 0) combinedStage.value = epochs[0];
        } catch (err) {
            console.error('Failed to load stages:', err);
            document.querySelector('main .card').innerHTML = '<p>No combined products available.</p>';
        }
    }

    async function loadRuns() {
        try {
            const runs = await API.getRuns();
            combinedRuns.innerHTML = '';
            runs.forEach(r => {
                const opt = document.createElement('option');
                opt.value = r.run_id;
                opt.textContent = `Run ${r.run_id}`;
                opt.selected = true; // Select all by default
                combinedRuns.appendChild(opt);
            });
        } catch (err) {
            console.error('Failed to load runs:', err);
        }
    }

    async function render() {
        const epoch = +combinedStage.value;
        if (!Number.isFinite(epoch)) return;

        const container = d3.select('#combinedList');
        const scalingInfo = document.getElementById('scalingInfo');

        // Get selected run IDs from multiselect
        const selectedRunIds = Array.from(combinedRuns.selectedOptions).map(opt => +opt.value);
        if (selectedRunIds.length === 0) {
            container.selectAll('*').remove();
            if (scalingInfo) scalingInfo.innerHTML = '';
            return;
        }

        const res = await API.getCombinedForEpoch(epoch);
        const runs = res.filter(r => r.snapshot && r.snapshot.grid_values && selectedRunIds.includes(r.run_id));

        // Display scaling information
        if (scalingInfo) {
            if (runs.length === 0) {
                scalingInfo.innerHTML = '<strong>No data available for selected runs at this stage</strong>';
            } else {
                const scalingRows = runs.map(r => {
                    const scalingValue = r.scaling !== undefined && r.scaling !== null
                        ? r.scaling.toFixed(6)
                        : 'N/A';
                    const energyValue = r.energy !== undefined && r.energy !== null
                        ? r.energy.toFixed(6)
                        : 'N/A';
                    return `<div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                        <span><strong>Run ${r.run_id}:</strong></span>
                        <span>Scaling: ${scalingValue} | Energy: ${energyValue}</span>
                    </div>`;
                }).join('');
                scalingInfo.innerHTML = `<div style="margin-bottom: 0.5rem;"><strong>Stage ${epoch} — scaling &amp; energy:</strong></div>${scalingRows}`;
            }
        }

        if (runs.length === 0) {
            container.selectAll('*').remove();
            return;
        }

        const nDims = runs.reduce((m, r) => Math.max(m, r.snapshot.grid_values.length), 0);
        container.selectAll('*').remove();
        if (!nDims) return;
        const cols = Math.max(1, Math.ceil(Math.sqrt(nDims)));
        container.style('grid-template-columns', `repeat(${cols}, minmax(300px, 1fr))`);

        const compBounds = {};
        for (let c = 0; c < nDims; c++) {
            let mins = [], maxs = [];
            runs.forEach(r => {
                const intervals = (r.snapshot.intervals || [])[c] || [];
                intervals.forEach(ib => {
                    const a = UI.toFloat(ib[0]);
                    const b = UI.toFloat(ib[1]);
                    if (isFinite(a)) mins.push(a);
                    if (isFinite(b)) maxs.push(b);
                });
            });
            let lo = -1, hi = 1;
            if (mins.length && maxs.length) {
                lo = Math.min(...mins), hi = Math.max(...maxs);
                if (!(isFinite(lo) && isFinite(hi)) || lo === hi) { lo = -1; hi = 1; }
            }
            const span = hi - lo; const margin = span > 0 ? 0.05 * span : 1.0;
            compBounds[c] = { min: lo - margin, max: hi + margin };
        }

        // Build a color map per run for consistent highlighting
        const palette = (Charts.__theme && Charts.__theme.palette) || [];
        const runColors = {};
        runs.forEach((r, i) => { runColors[`run_${r.run_id}`] = palette[i % (palette.length || 7)] || '#555'; });

        // Use a reasonable default for max intervals (400)
        const maxInt = 400;

        for (let c = 0; c < nDims; c++) {
            const el = document.createElement('div');
            el.className = 'chart';
            container.node().appendChild(el);
            const comps = runs.map((r, idx) => {
                const vals = (r.snapshot.grid_values || [])[c] || [];
                const ilist = (r.snapshot.intervals || [])[c] || [];
                const stride = Math.max(1, Math.ceil(vals.length / maxInt));
                const intervals = ilist.map((ib, i) => [UI.toFloat(ib[0]), UI.toFloat(ib[1]), UI.toFloat(vals[i])]).filter(v => v.length === 3).filter((_, i) => i % stride === 0);
                return { intervals, key: `run_${r.run_id}`, color: runColors[`run_${r.run_id}`] };
            });
            Charts.stepFunctions(el, comps, compBounds[c], { title: `Component ${c} (Stage ${epoch})`, hoverable: true });

            // Hover highlight + tooltip like Tree Evolution
            const svg = d3.select(el).select('svg');
            const tooltip = Charts.getTooltip();
            svg.on('mousemove', function (event) {
                const target = event.target;
                if (!target || target.tagName !== 'path') return;
                const key = target.getAttribute('data-key');
                if (!key) return;
                d3.selectAll('#combinedList .chart svg .trace')
                    .attr('opacity', function () { return this.getAttribute('data-key') === key ? 1.0 : 0.15; })
                    .attr('stroke-width', function () { return this.getAttribute('data-key') === key ? 2.0 : 1.0; });
                const runIdDisp = key.replace('run_', '');
                const color = runColors[key] || '#444';
                tooltip.style.display = 'block';
                tooltip.style.left = (event.pageX + 12) + 'px';
                tooltip.style.top = (event.pageY + 12) + 'px';
                tooltip.innerHTML = `<div style="display:flex;align-items:center;gap:8px"><span style="display:inline-block;width:10px;height:10px;background:${color};border-radius:2px"></span><b>Run ${runIdDisp}</b></div><div>Component ${c}</div>`;
            });
            svg.on('mouseleave', function () {
                d3.selectAll('#combinedList .chart svg .trace')
                    .attr('opacity', 1.0)
                    .attr('stroke-width', 1.4);
                const t = Charts.getTooltip();
                if (t) t.style.display = 'none';
            });
        }
    }

    // Set up event listeners
    combinedStage.addEventListener('change', render);
    combinedRuns.addEventListener('change', render);

    // Initialize
    await loadEpochs();
    await loadRuns();
    await render();
})();
