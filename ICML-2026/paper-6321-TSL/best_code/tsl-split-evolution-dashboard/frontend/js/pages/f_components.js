(async function () {
    const runSelect = document.getElementById('runSelect');
    const epochSelect = document.getElementById('epochSelect');
    const treeSelect = document.getElementById('treeSelect');
    const iterationRange = document.getElementById('iterationRange');
    const iterationValue = document.getElementById('iterationValue');
    const axesGrid = document.getElementById('axesGrid');
    const mergedToggle = document.getElementById('mergedToggle');
    const logScaleToggle = document.getElementById('logScaleToggle');
    const lambdaGradientToggle = document.getElementById('lambdaGradientToggle');
    const highlightChoiceToggle = document.getElementById('highlightChoiceToggle');

    let currentRunId = null;
    let epochsTrees = {};
    const timelineCache = new Map();
    const layoutManager = ComponentPageShared.createLayoutManager(axesGrid);
    const renderScheduler = ComponentPageShared.createRenderScheduler(() => render());
    const renderSeqTracker = ComponentPageShared.createRenderSequence();
    const lambdaCache = ComponentPageShared.createLambdaCache();

    const showStatus = (message) => ComponentPageShared.showStatus(axesGrid, layoutManager, message);
    const getSelectedTrees = () => ComponentPageShared.getSelectedTrees(treeSelect);

    function ensureLayout(layoutKey, merged, columnsMeta) {
        return layoutManager.ensure({
            layoutKey,
            modeKey: merged ? 'merged' : 'single',
            columnsMeta,
            setupGrid(gridEl, mode) {
                if (mode === 'merged') {
                    gridEl.style.display = 'grid';
                    gridEl.style.gridTemplateColumns = '1fr 1fr';
                    gridEl.style.gap = '20px';
                } else {
                    const nAxes = columnsMeta.length;
                    const cols = Math.max(2, Math.min(4, Math.ceil(Math.sqrt(nAxes || 1))));
                    gridEl.style.display = 'grid';
                    gridEl.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
                    gridEl.style.gap = '20px';
                }
            },
            createEntry(col, mode, ctx) {
                const { axesGrid, createChartContainer } = ctx;
                if (mode === 'merged') {
                    const axisLabel = document.createElement('div');
                    axisLabel.style.gridColumn = '1 / -1';
                    axisLabel.style.textAlign = 'center';
                    axisLabel.style.fontWeight = 'bold';
                    axisLabel.style.padding = '10px';
                    axisLabel.style.backgroundColor = '#f3f4f6';
                    axisLabel.style.borderRadius = '4px';
                    axisLabel.textContent = `Axis ${col}`;
                    axesGrid.appendChild(axisLabel);

                    const fPlusContainer = createChartContainer();
                    axesGrid.appendChild(fPlusContainer);
                    const fMinusContainer = createChartContainer();
                    axesGrid.appendChild(fMinusContainer);

                    return {
                        col,
                        labelEl: axisLabel,
                        fPlusContainer,
                        fMinusContainer
                    };
                }

                const chartContainer = createChartContainer();
                axesGrid.appendChild(chartContainer);
                return {
                    col,
                    chartContainer
                };
            }
        });
    }

    function calculateTreeMaxIters(timeline, epoch, treeIds) {
        if (window.UI && typeof window.UI.calculateTreeMaxIters === 'function') {
            return window.UI.calculateTreeMaxIters(timeline, epoch, treeIds);
        }
        const result = new Map();
        const wanted = Array.isArray(treeIds) ? treeIds.map(Number).filter(Number.isFinite) : [];
        const wantedSet = new Set(wanted);
        wanted.forEach(t => result.set(t, 0));
        if (!Array.isArray(timeline) || !Number.isFinite(+epoch) || wantedSet.size === 0) return result;
        const eTarget = Number(epoch);
        for (const ev of timeline) {
            if (Number(ev?.epoch) !== eTarget) continue;
            const t = Number(ev?.tree_id);
            if (!wantedSet.has(t)) continue;
            const it = Number(ev?.iter_no) || 0;
            const prev = result.get(t) || 0;
            if (it > prev) result.set(t, it);
        }
        return result;
    }

    async function ensureLambdaValues(runId, epoch, treeId, treeMaxIter, fetchFn, fallbackPlus = null, fallbackMinus = null) {
        return lambdaCache.ensure(
            runId,
            epoch,
            treeId,
            treeMaxIter,
            fetchFn,
            fallbackPlus,
            fallbackMinus
        );
    }

    async function getRunTimeline(runId) {
        if (!timelineCache.has(runId)) {
            const tl = await API.getTimeline(runId);
            timelineCache.set(runId, Array.isArray(tl) ? tl : []);
        }
        return timelineCache.get(runId);
    }

    async function updateIterationBounds() {
        const runId = +runSelect.value;
        const epoch = +epochSelect.value;
        const merged = !!mergedToggle?.checked;
        const selectedTrees = getSelectedTrees();

        if (![runId, epoch].every(Number.isFinite)) return;
        if (!merged && selectedTrees.length === 0) return;

        const tl = await getRunTimeline(runId);
        let relevant = tl.filter(e => Number(e.epoch) === epoch);
        if (!merged) {
            const target = selectedTrees[0];
            relevant = relevant.filter(e => Number(e.tree_id) === target);
        } else if (selectedTrees.length > 0) {
            const allowed = new Set(selectedTrees);
            relevant = relevant.filter(e => allowed.has(Number(e.tree_id)));
        }
        const maxIter = relevant.length ? Math.max(...relevant.map(e => Number(e.iter_no) || 0)) : 0;

        iterationRange.max = String(Math.max(0, maxIter));
        const curVal = Number(iterationRange.value) || 0;
        // If current slider value is out of range, clamp it. If it's still the default 0
        // and there are available iterations, default the slider to the max iteration so we
        // don't accidentally fetch the initial (iter 0) state for many trees.
        if (curVal > maxIter) {
            iterationRange.value = String(maxIter);
            iterationValue.textContent = String(maxIter);
        } else if (curVal === 0 && maxIter > 0) {
            iterationRange.value = String(maxIter);
            iterationValue.textContent = String(maxIter);
        }
    }

    async function loadRuns() {
        try {
            const runs = await API.getRuns();
            runSelect.innerHTML = '';
            runs.forEach(r => {
                const opt = document.createElement('option');
                opt.value = r.run_id;
                opt.textContent = `Run ${r.run_id}`;
                runSelect.appendChild(opt);
            });
            if (runs.length > 0) {
                // Pick the first run that has f-component epochs/trees if possible
                let picked = false;
                for (const r of runs) {
                    runSelect.value = r.run_id;
                    try {
                        const et = await API.getFComponentEpochsTrees(r.run_id);
                        if (et && Object.keys(et).length > 0) {
                            picked = true;
                            break;
                        }
                    } catch (e) {
                        // fallback to general epochs_trees in loadEpochsTrees
                        try {
                            const et2 = await API.getEpochsTrees(r.run_id);
                            if (et2 && Object.keys(et2).length > 0) {
                                picked = true;
                                break;
                            }
                        } catch (e2) { }
                    }
                }
                if (!picked) runSelect.value = runs[0].run_id;
                await loadEpochsTrees();
            }
        } catch (err) {
            console.error('Failed to load runs:', err);
        }
    }

    async function loadEpochsTrees() {
        const runId = parseInt(runSelect.value);
        if (!runId) return;

        try {
            // Try f_component_epochs_trees first to only show epochs/trees that have f_component_stats data
            let epochsTreesData;
            try {
                epochsTreesData = await API.getFComponentEpochsTrees(runId);
            } catch (err) {
                console.warn('f_component_epochs_trees failed, falling back to epochs_trees:', err);
                epochsTreesData = await API.getEpochsTrees(runId);
            }

            epochsTrees = epochsTreesData;
            currentRunId = runId;

            const epochs = Object.keys(epochsTrees).map(Number).sort((a, b) => a - b);
            epochSelect.innerHTML = '';
            epochs.forEach(epoch => {
                const opt = document.createElement('option');
                opt.value = epoch;
                opt.textContent = `Stage ${epoch}`;
                epochSelect.appendChild(opt);
            });

            if (epochs.length > 0) {
                epochSelect.value = epochs[0];
                await refreshTreeList();
                await updateIterationBounds();
            }
        } catch (err) {
            console.error('Failed to load epochs/trees:', err);
            epochSelect.innerHTML = '<option value="">Error loading data</option>';
            treeSelect.innerHTML = '<option value="">Error loading data</option>';
        }
    }

    async function refreshTreeList() {
        const runId = parseInt(runSelect.value);
        if (isNaN(runId)) return;

        const epoch = parseInt(epochSelect.value);
        if (isNaN(epoch)) return;

        const epochKey = String(epoch);
        if (!epochsTrees[epochKey]) return;

        const trees = epochsTrees[epochKey].sort((a, b) => a - b);
        const prevSelected = new Set(getSelectedTrees());
        treeSelect.innerHTML = '';
        let anySelected = false;
        trees.forEach(tree => {
            const opt = document.createElement('option');
            opt.value = tree;
            opt.textContent = `Product ${tree}`;
            if (prevSelected.has(tree)) {
                opt.selected = true;
                anySelected = true;
            }
            treeSelect.appendChild(opt);
        });

        if (!anySelected && treeSelect.options.length > 0) {
            treeSelect.options[0].selected = true;
        }

        if (treeSelect.options.length > 0) {
            await updateIterationBounds();
            await render();
        }
    }

    function getValueFromIntervals(intervals, xVal, bounds) {
        if (!intervals || intervals.length === 0) return null;

        for (const [a, b, v] of intervals) {
            const left = Number.isFinite(a) ? a : bounds.min;
            const right = Number.isFinite(b) ? b : bounds.max;
            if (xVal >= left && xVal < right) return v;
        }

        // Return last interval value if past the end
        if (intervals.length > 0) {
            return intervals[intervals.length - 1][2];
        }
        return null;
    }

    function formatLambdaTooltip(lambdaInfo) {
        if (!lambdaInfo) return '';
        const parts = [];
        const lp = Number(lambdaInfo.lambda_plus);
        const lm = Number(lambdaInfo.lambda_minus);
        if (Number.isFinite(lp)) parts.push(`λ⁺=${lp.toPrecision(6)}`);
        if (Number.isFinite(lm)) parts.push(`λ⁻=${lm.toPrecision(6)}`);
        return parts.length ? `<div>${parts.join(' · ')}</div>` : '';
    }

    function createHoverInteractions(svg, tooltip, bounds, comps, axisCol, containerEl) {
        svg.on('mousemove', function (event) {
            const target = event.target;
            if (!target || target.tagName !== 'path') return;
            const key = target.getAttribute('data-key');
            if (!key) return;

            // Extract tree ID if this is merged mode (key format: f_plus_tree_X or f_minus_tree_X)
            let treeId = null;
            let highlightPattern = key;
            if (key.includes('_tree_')) {
                treeId = key.split('_tree_')[1];
                highlightPattern = key;
            }

            // Highlight traces across all charts in axesGrid
            if (treeId) {
                // Merged mode: highlight this tree across all axes (both f+ and f-)
                d3.selectAll('#axesGrid .chart svg .trace')
                    .attr('opacity', function () {
                        const traceKey = this.getAttribute('data-key');
                        return (traceKey && traceKey.includes(`_tree_${treeId}`)) ? 1.0 : 0.15;
                    })
                    .attr('stroke-width', function () {
                        const traceKey = this.getAttribute('data-key');
                        return (traceKey && traceKey.includes(`_tree_${treeId}`)) ? 2.0 : 1.0;
                    });
            } else {
                // Single tree mode: only highlight in current chart
                const chartSvg = d3.select(containerEl);
                chartSvg.selectAll('.trace')
                    .attr('opacity', function () {
                        return this.getAttribute('data-key') === key ? 1.0 : 0.15;
                    })
                    .attr('stroke-width', function () {
                        return this.getAttribute('data-key') === key ? 2.0 : 1.0;
                    });
            }

            // Find the component for this key
            const comp = comps.find(c => (c.key || '') === key);
            if (!comp) return;

            // Determine label
            const baseLabel = comp.seriesLabel || (key.includes('f_minus') ? 'f-' : 'f+');
            let titleLabel = baseLabel;
            if (key.includes('_tree_')) {
                titleLabel = `${baseLabel} (product ${treeId})`;
            }
            const color = comp.color || '#3b82f6';

            // Compute value at cursor
            const { innerW } = Charts.sizeOf(svg.node().parentElement);
            const [mx, my] = d3.pointer(event, svg.select('g').node());
            const x = d3.scaleLinear().domain([bounds.min, bounds.max]).range([0, innerW]);
            const xVal = x.invert(mx);
            const valueAtCursor = getValueFromIntervals(comp.intervals, xVal, bounds);
            const lambdaHtml = formatLambdaTooltip(comp.lambdaInfo);

            // Show tooltip
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 12) + 'px';
            tooltip.style.top = (event.pageY + 12) + 'px';
            const valStr = (valueAtCursor != null && isFinite(+valueAtCursor)) ? Number(valueAtCursor).toPrecision(6) : '—';

            const header = comps.length > 1 || treeId
                ? `<div style="display:flex;align-items:center;gap:8px"><span style="display:inline-block;width:10px;height:10px;background:${color};border-radius:2px"></span><b>${titleLabel}</b></div>`
                : `<b>${titleLabel} (Axis ${axisCol})</b>`;
            tooltip.innerHTML = `${header}<div>Axis ${axisCol}</div><div>value: ${valStr}</div>${lambdaHtml}`;
        });

        svg.on('mouseleave', function () {
            // Reset all traces across all charts
            d3.selectAll('#axesGrid .chart svg .trace')
                .attr('opacity', 1.0)
                .attr('stroke-width', 1.4);
            const t = Charts.getTooltip();
            if (t) t.style.display = 'none';
        });
    }

    function renderAxisChart(container, axisData, isFPlus, useLogScale = null, lambdaPlus = null, lambdaMinus = null, colorScalePlus = null, colorScaleMinus = null) {
        if (typeof Charts === 'undefined') {
            container.innerHTML = '<p>Error: Charts module not loaded. Please refresh the page.</p>';
            console.error('Charts is not defined. Make sure charts.js is loaded before f_components.js');
            return;
        }

        // Ensure container has proper sizing
        if (container.style.width === '' || container.style.width === 'auto') {
            const rect = container.getBoundingClientRect();
            if (rect.width > 0) {
                container.style.width = `${rect.width}px`;
            }
        }

        const intervals = isFPlus ? (axisData.intervals_plus || []) : (axisData.intervals_minus || []);
        const title = isFPlus ? `f+ (Axis ${axisData.col})` : `f- (Axis ${axisData.col})`;

        if (!intervals || intervals.length === 0) {
            container.innerHTML = '<p>No data</p>';
            return;
        }

        const bounds = UI.computeBoundsFromIntervalsTriples(intervals);

        // Determine color based on toggle
        const useLambdaGradient = lambdaGradientToggle ? lambdaGradientToggle.checked : false;
        let lineColor = isFPlus ? '#3b82f6' : '#ef4444'; // Default colors
        if (useLambdaGradient) {
            if (isFPlus && colorScalePlus && lambdaPlus != null && isFinite(lambdaPlus)) {
                lineColor = colorScalePlus(lambdaPlus);
            } else if (!isFPlus && colorScaleMinus && lambdaMinus != null && isFinite(lambdaMinus)) {
                lineColor = colorScaleMinus(lambdaMinus);
            }
        }

        // Format intervals for Charts.stepFunctions
        const lambdaInfo = {
            lambda_plus: lambdaPlus,
            lambda_minus: lambdaMinus
        };
        const comps = [{
            intervals: intervals,
            key: isFPlus ? 'f_plus' : 'f_minus',
            color: lineColor,
            seriesLabel: isFPlus ? 'f+' : 'f-',
            lambdaInfo
        }];

        // Double-check Charts is available before calling
        if (typeof Charts === 'undefined' || typeof Charts.stepFunctions !== 'function') {
            container.innerHTML = '<p>Error: Charts library not loaded. Please refresh the page.</p>';
            console.error('Charts.stepFunctions is not available');
            return;
        }

        // Determine log scale: use toggle if provided, otherwise default to true (log scale)
        const useLog = useLogScale !== null ? useLogScale : true;

        try {
            Charts.stepFunctions(container, comps, bounds, {
                title: title,
                log: useLog,
                hoverable: true
            });

            // Add hover interactions
            const svg = d3.select(container).select('svg');
            const tooltip = Charts.getTooltip();
            if (svg.node() && tooltip) {
                createHoverInteractions(svg, tooltip, bounds, comps, axisData.col, container);
            }
        } catch (err) {
            container.innerHTML = `<p>Error rendering chart: ${err.message}</p>`;
            console.error('Error in Charts.stepFunctions:', err);
        }
    }

    function renderSingleTreeAxisChart(container, axisData, useLogScale, colorScalePlus, colorScaleMinus, treeData, isSelected = false, isBest = false) {
        const intervalsPlus = axisData.intervals_plus || [];
        const intervalsMinus = axisData.intervals_minus || [];

        if ((!intervalsPlus || intervalsPlus.length === 0) && (!intervalsMinus || intervalsMinus.length === 0)) {
            container.innerHTML = '<p>No data</p>';
            return;
        }

        const bounds = UI.computeBoundsFromManyIntervalsTriples([intervalsPlus, intervalsMinus]);
        const comps = [];

        let colorPlus = '#3b82f6';
        let colorMinus = '#ef4444';
        if (colorScalePlus && treeData && treeData.lambda_plus != null && isFinite(treeData.lambda_plus)) {
            colorPlus = colorScalePlus(treeData.lambda_plus);
        }
        if (colorScaleMinus && treeData && treeData.lambda_minus != null && isFinite(treeData.lambda_minus)) {
            colorMinus = colorScaleMinus(treeData.lambda_minus);
        }
        // Best takes precedence (yellow), otherwise selected candidates are green
        if (isBest) {
            colorPlus = '#f59e0b';
            colorMinus = '#f59e0b';
        } else if (isSelected) {
            colorPlus = '#10b981';
            colorMinus = '#10b981';
        }

        const lambdaInfo = {
            lambda_plus: treeData?.lambda_plus ?? null,
            lambda_minus: treeData?.lambda_minus ?? null
        };

        if (intervalsPlus && intervalsPlus.length > 0) {
            comps.push({
                intervals: intervalsPlus,
                key: 'f_plus',
                color: colorPlus,
                seriesLabel: 'f+',
                lambdaInfo
            });
        }

        if (intervalsMinus && intervalsMinus.length > 0) {
            comps.push({
                intervals: intervalsMinus,
                key: 'f_minus',
                color: colorMinus,
                seriesLabel: 'f-',
                lambdaInfo
            });
        }

        if (typeof Charts !== 'undefined' && typeof Charts.stepFunctions === 'function') {
            Charts.stepFunctions(container, comps, bounds, {
                title: `Axis ${axisData.col} (f+ and f-)`,
                log: useLogScale,
                hoverable: true
            });
            const svg = d3.select(container).select('svg');
            const tooltip = Charts.getTooltip();
            if (svg.node() && tooltip) {
                createHoverInteractions(svg, tooltip, bounds, comps, axisData.col, container);
            }
        } else {
            container.innerHTML = '<p>Error: Charts library not loaded.</p>';
        }
    }

    function renderMergedChart(container, allTreesData, isFPlus, axisCol, useLogScale = null, lambdaDataMap = null, colorScalePlus = null, colorScaleMinus = null, candidatesSet = null, bestId = null) {
        if (typeof Charts === 'undefined') {
            container.innerHTML = '<p>Error: Charts module not loaded. Please refresh the page.</p>';
            console.error('Charts is not defined. Make sure charts.js is loaded before f_components.js');
            return;
        }

        // Ensure container has proper sizing
        if (container.style.width === '' || container.style.width === 'auto') {
            const rect = container.getBoundingClientRect();
            if (rect.width > 0) {
                container.style.width = `${rect.width}px`;
            }
        }

        // allTreesData is an array of { treeId, intervals_plus, intervals_minus }
        if (!allTreesData || allTreesData.length === 0) {
            container.innerHTML = '<p>No data</p>';
            return;
        }

        // Get the appropriate intervals (f+ or f-) from all trees
        const intervalsToUse = isFPlus ? 'intervals_plus' : 'intervals_minus';

        const bounds = UI.computeBoundsFromManyIntervalsTriples(
            allTreesData.map(td => td[intervalsToUse] || [])
        );

        // Setup tree colors using palette or gradient
        const useLambdaGradient = lambdaGradientToggle ? lambdaGradientToggle.checked : false;
        const palette = (Charts.__theme && Charts.__theme.palette) || ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4'];

        // Format intervals for Charts.stepFunctions: all trees for this component type
        const comps = [];
        allTreesData.forEach((treeData, treeIdx) => {
            const treeId = treeData.treeId;
            let treeColor = palette[treeIdx % palette.length];
            const lambdaData = lambdaDataMap ? lambdaDataMap.get(treeId) : null;

            if (useLambdaGradient && lambdaData) {
                if (isFPlus && colorScalePlus) {
                    const lambdaVal = lambdaData.lambda_plus;
                    if (lambdaVal != null && isFinite(lambdaVal)) {
                        treeColor = colorScalePlus(lambdaVal);
                    } else {
                        console.warn(`Tree ${treeId} f+: invalid lambda_plus value:`, lambdaVal);
                    }
                } else if (!isFPlus && colorScaleMinus) {
                    const lambdaVal = lambdaData.lambda_minus;
                    if (lambdaVal != null && isFinite(lambdaVal)) {
                        treeColor = colorScaleMinus(lambdaVal);
                    } else {
                        console.warn(`Tree ${treeId} f-: invalid lambda_minus value:`, lambdaVal);
                    }
                } else {
                    console.warn(`Tree ${treeId}: color scale not available (isFPlus=${isFPlus}, colorScalePlus=${!!colorScalePlus}, colorScaleMinus=${!!colorScaleMinus})`);
                }
            }
            // Highlight trees selected by the combination choice with green
            try {
                const tidNum = Number(treeId);
                if ((candidatesSet && candidatesSet.has(tidNum)) || (bestId !== null && bestId === tidNum)) {
                    treeColor = '#10b981';
                }
            } catch (e) {
                // ignore
            }

            const intervals = treeData[intervalsToUse] || [];

            if (intervals.length > 0) {
                comps.push({
                    intervals: intervals,
                    key: `${isFPlus ? 'f_plus' : 'f_minus'}_tree_${treeId}`,
                    color: treeColor,
                    split_label: `T${treeId}`,
                    seriesLabel: isFPlus ? 'f+' : 'f-',
                    lambdaInfo: lambdaData || null
                });
            }
        });

        if (comps.length === 0) {
            container.innerHTML = '<p>No data</p>';
            return;
        }

        // Double-check Charts is available before calling
        if (typeof Charts === 'undefined' || typeof Charts.stepFunctions !== 'function') {
            container.innerHTML = '<p>Error: Charts library not loaded. Please refresh the page.</p>';
            console.error('Charts.stepFunctions is not available');
            return;
        }

        // Determine log scale: use toggle if provided, otherwise default to true (log scale)
        const useLog = useLogScale !== null ? useLogScale : true;

        try {
            const title = isFPlus
                ? `f+ (Axis ${axisCol}, Merged)`
                : `f- (Axis ${axisCol}, Merged)`;
            Charts.stepFunctions(container, comps, bounds, {
                title: title,
                log: useLog,
                hoverable: true
            });

            // Add hover interactions
            const svg = d3.select(container).select('svg');
            const tooltip = Charts.getTooltip();
            if (svg.node() && tooltip) {
                createHoverInteractions(svg, tooltip, bounds, comps, axisCol, container);
            }
        } catch (err) {
            container.innerHTML = `<p>Error rendering chart: ${err.message}</p>`;
            console.error('Error in Charts.stepFunctions:', err);
        }
    }


    // Event listeners
    runSelect.addEventListener('change', loadEpochsTrees);
    epochSelect.addEventListener('change', async () => {
        await refreshTreeList();
        await updateIterationBounds();
    });
    treeSelect.addEventListener('change', async () => {
        await updateIterationBounds();
        await render();
    });
    iterationRange.addEventListener('input', (e) => {
        iterationValue.textContent = e.target.value;
        renderScheduler.schedule();
    });
    mergedToggle.addEventListener('change', async () => {
        await updateIterationBounds();
        render();
    });

    if (logScaleToggle) {
        logScaleToggle.addEventListener('change', () => {
            render();
        });
    }

    if (lambdaGradientToggle) {
        lambdaGradientToggle.addEventListener('change', async () => {
            try {
                const checked = !!lambdaGradientToggle.checked;
                if (checked) {
                    mergedToggle.checked = true;
                    const runId = parseInt(runSelect.value);
                    const epoch = parseInt(epochSelect.value);
                    if (runId && Number.isFinite(epoch)) {
                        const tl = await getRunTimeline(runId);
                        const relevant = tl.filter(e => Number(e.epoch) === epoch);
                        const globalMax = relevant.length ? Math.max(...relevant.map(e => Number(e.iter_no) || 0)) : 0;
                        iterationRange.max = String(Math.max(0, globalMax));
                        iterationRange.value = String(globalMax);
                        iterationValue.textContent = String(globalMax);
                    }
                }
            } catch (err) {
                console.warn('Error while handling lambdaGradientToggle change:', err);
            } finally {
                render();
            }
        });
    }
    if (highlightChoiceToggle) {
        highlightChoiceToggle.addEventListener('change', async () => {
            try {
                const checked = !!highlightChoiceToggle.checked;
                const runId = parseInt(runSelect.value);
                const epoch = parseInt(epochSelect.value);
                if (checked) {
                    // ensure merged mode
                    if (mergedToggle) mergedToggle.checked = true;
                    // refresh tree list so options are populated
                    await refreshTreeList();
                    // fetch combination choice and select candidate trees
                    try {
                        if (runId && Number.isFinite(epoch)) {
                            const combo = await API.getCombinationChoice(runId, epoch);
                            const ids = (combo && combo.candidates) ? combo.candidates.map(c => String(c.tree_id)) : [];
                            const best = combo && Number.isFinite(+combo.best_index) ? String(+combo.best_index) : null;
                            if (treeSelect && treeSelect.options) {
                                for (let i = 0; i < treeSelect.options.length; i++) {
                                    const opt = treeSelect.options[i];
                                    opt.selected = ids.includes(opt.value) || (best !== null && opt.value === best);
                                }
                            }
                        }
                    } catch (e) {
                        console.warn('Failed to fetch combination choice for highlight toggle:', e);
                    }
                    await updateIterationBounds();
                    render();
                } else {
                    // on uncheck, simply re-render (do not modify selection)
                    render();
                }
            } catch (err) {
                console.warn('Error handling highlightChoiceToggle change:', err);
            }
        });
    }

    // Wait for Charts to be available before initializing
    function waitForCharts(maxAttempts = 50) {
        return new Promise((resolve, reject) => {
            let attempts = 0;
            const checkCharts = () => {
                if (typeof Charts !== 'undefined') {
                    resolve();
                } else if (attempts < maxAttempts) {
                    attempts++;
                    setTimeout(checkCharts, 100);
                } else {
                    reject(new Error('Charts library failed to load'));
                }
            };
            checkCharts();
        });
    }

    // Initialize
    try {
        await waitForCharts();
        await loadRuns();
    } catch (err) {
        console.error('Failed to initialize:', err);
        axesGrid.innerHTML = `<p>Error: ${err.message}</p>`;
    }
    async function render() {
        const seq = renderSeqTracker.next();
        const runId = parseInt(runSelect.value);
        const epoch = parseInt(epochSelect.value);
        const iteration = parseInt(iterationRange.value);
        const selectedTrees = getSelectedTrees();
        const merged = !!mergedToggle?.checked;
        const selectionKey = selectedTrees.slice().sort((a, b) => a - b).join(',') || 'all';
        const layoutKey = merged
            ? `merged:${runId}:${epoch}:${selectionKey}`
            : `single:${runId}:${epoch}:${selectionKey}`;

        // Prepare combination choice placeholders shared across merged/single views
        let comboChoice = null;
        let candidatesSet = new Set();
        let bestId = null;

        if (merged) {
            if (![runId, epoch, iteration].every(Number.isFinite)) {
                showStatus('Please select run, epoch, and iteration');
                return;
            }
        } else {
            if (![runId, epoch, iteration].every(Number.isFinite)) {
                showStatus('Please select run, epoch, tree, and iteration');
                return;
            }
            if (selectedTrees.length !== 1) {
                showStatus('Please select exactly one tree when merged is off');
                return;
            }
        }

        if (!layoutManager.hasLayout()) {
            axesGrid.innerHTML = '<p>Loading...</p>';
            axesGrid.style.display = 'block';
        }

        try {
            if (merged) {
                const epochKey = String(epoch);
                const allTrees = (epochsTrees[epochKey] || []).sort((a, b) => a - b);
                if (allTrees.length === 0) {
                    showStatus('No trees available for this epoch');
                    return;
                }

                const targetTrees = selectedTrees.length
                    ? allTrees.filter(t => selectedTrees.includes(t))
                    : allTrees;
                if (targetTrees.length === 0) {
                    showStatus('Selected trees are not available for this epoch');
                    return;
                }

                const tl = await getRunTimeline(runId);
                if (!renderSeqTracker.isCurrent(seq)) return;
                const treeMaxIters = calculateTreeMaxIters(tl, epoch, targetTrees);

                let allTreesData = null;
                try {
                    const multiKey = UI.Cache.makeKey('f_per_axis_multi', runId, epoch, iteration, targetTrees.join(','));
                    const multi = await UI.Cache.fetchOrGet(
                        multiKey,
                        () => API.getFComponentPerAxisMulti(runId, epoch, iteration, targetTrees)
                    );
                    if (!renderSeqTracker.isCurrent(seq)) return;
                    if (multi && Array.isArray(multi.trees)) {
                        allTreesData = multi.trees.map(t => {
                            const tId = Number(t.tree_id);
                            const treeMaxIter = treeMaxIters.get(tId) || 0;
                            return {
                                treeId: tId,
                                data: {
                                    axes: t.axes || [],
                                    lambda_plus: t.lambda_plus,
                                    lambda_minus: t.lambda_minus
                                },
                                treeMaxIter
                            };
                        });
                    }
                } catch (e) {
                    console.warn('Multi-tree f+/f- fetch failed; falling back to per-tree requests:', e);
                }

                if (!allTreesData) {
                    allTreesData = await Promise.all(
                        targetTrees.map(async (tId) => {
                            const treeMaxIter = treeMaxIters.get(tId) || 0;
                            const iterToUse = Math.min(iteration, treeMaxIter);
                            const perAxisKey = UI.Cache.makeKey('f_per_axis', runId, epoch, tId, iterToUse);
                            try {
                                const data = await UI.Cache.fetchOrGet(perAxisKey, () =>
                                    API.getFComponentPerAxis(runId, epoch, tId, iterToUse)
                                );
                                return { treeId: tId, data, treeMaxIter };
                            } catch (err) {
                                console.warn(`Failed to fetch data for tree ${tId} at iter ${iterToUse}:`, err);
                                return { treeId: tId, data: null, treeMaxIter };
                            }
                        })
                    );
                    if (!renderSeqTracker.isCurrent(seq)) return;
                }

                const firstTreeWithData = allTreesData.find(t => t.data && t.data.axes && t.data.axes.length > 0);
                if (!firstTreeWithData) {
                    showStatus('No data available for selected trees');
                    return;
                }

                const columnsMeta = firstTreeWithData.data.axes.map(a => a.col);
                const axisEntries = ensureLayout(layoutKey, true, columnsMeta);
                const axisMap = new Map(axisEntries.map(entry => [entry.col, entry]));

                const useLambdaGradient = lambdaGradientToggle ? lambdaGradientToggle.checked : false;
                const lambdaDataMap = new Map();
                const lambdaPlusValues = [];
                const lambdaMinusValues = [];
                const lambdaResults = await Promise.all(
                    allTreesData.map(async ({ treeId, data, treeMaxIter }) => {
                        const fallbackPlus = data?.lambda_plus ?? null;
                        const fallbackMinus = data?.lambda_minus ?? null;
                        let vals = {
                            lambda_plus: fallbackPlus,
                            lambda_minus: fallbackMinus
                        };
                        if (vals.lambda_plus == null && vals.lambda_minus == null) {
                            vals = await ensureLambdaValues(
                                runId,
                                epoch,
                                treeId,
                                treeMaxIter,
                                API.getFComponentPerAxis,
                                fallbackPlus,
                                fallbackMinus
                            );
                        }
                        return { treeId, vals };
                    })
                );
                if (!renderSeqTracker.isCurrent(seq)) return;
                lambdaResults.forEach(({ treeId, vals }) => {
                    lambdaDataMap.set(treeId, vals);
                    const lp = vals.lambda_plus;
                    const lm = vals.lambda_minus;
                    if (useLambdaGradient) {
                        if (lp != null && isFinite(lp)) lambdaPlusValues.push(lp);
                        if (lm != null && isFinite(lm)) lambdaMinusValues.push(lm);
                    }
                });

                const { colorScalePlus, colorScaleMinus } = useLambdaGradient
                    ? UI.createLambdaColorScales(lambdaPlusValues, lambdaMinusValues)
                    : { colorScalePlus: null, colorScaleMinus: null };
                // Prepare combination choice info only if highlight toggle is enabled
                // (we already declared placeholders above)
                if (highlightChoiceToggle && highlightChoiceToggle.checked) {
                    try {
                        comboChoice = await API.getCombinationChoice(runId, epoch);
                    } catch (e) {
                        comboChoice = null;
                    }
                    candidatesSet = new Set((comboChoice && comboChoice.candidates) ? comboChoice.candidates.map(c => Number(c.tree_id)) : []);
                    bestId = comboChoice && Number.isFinite(+comboChoice.best_index) ? +comboChoice.best_index : null;
                }

                for (let axisIdx = 0; axisIdx < columnsMeta.length; axisIdx++) {
                    if (!renderSeqTracker.isCurrent(seq)) return;
                    const col = columnsMeta[axisIdx];
                    const entry = axisMap.get(col);
                    if (!entry) continue;

                    const allTreesAxisData = [];
                    allTreesData.forEach(({ treeId, data }) => {
                        if (!data || !data.axes) return;
                        const axisData = data.axes.find(a => a.col === col) || data.axes[axisIdx];
                        if (!axisData) return;

                        allTreesAxisData.push({
                            treeId,
                            intervals_plus: axisData.intervals_plus || [],
                            intervals_minus: axisData.intervals_minus || []
                        });
                    });

                    if (allTreesAxisData.length === 0) {
                        entry.fPlusContainer.innerHTML = '<p>No data</p>';
                        entry.fMinusContainer.innerHTML = '<p>No data</p>';
                        continue;
                    }

                    const useLogScale = logScaleToggle ? logScaleToggle.checked : null;
                    renderMergedChart(entry.fPlusContainer, allTreesAxisData, true, col, useLogScale, lambdaDataMap, colorScalePlus, colorScaleMinus, candidatesSet, bestId);
                    renderMergedChart(entry.fMinusContainer, allTreesAxisData, false, col, useLogScale, lambdaDataMap, colorScalePlus, colorScaleMinus, candidatesSet, bestId);
                }
            } else {
                const treeId = selectedTrees[0];
                const tl = await getRunTimeline(runId);
                if (!renderSeqTracker.isCurrent(seq)) return;
                const treeMaxIters = calculateTreeMaxIters(tl, epoch, [treeId]);
                const treeMaxIter = treeMaxIters.get(treeId) || 0;
                const iterToUse = Math.min(iteration, treeMaxIter);
                const perAxisKey = UI.Cache.makeKey('f_per_axis', runId, epoch, treeId, iterToUse);
                const data = await UI.Cache.fetchOrGet(perAxisKey, () =>
                    API.getFComponentPerAxis(runId, epoch, treeId, iterToUse)
                );
                if (!renderSeqTracker.isCurrent(seq)) return;

                if (!data.axes || data.axes.length === 0) {
                    showStatus('No data available');
                    return;
                }

                const columnsMeta = data.axes.map(a => a.col);
                const axisEntries = ensureLayout(layoutKey, false, columnsMeta);
                const axisMap = new Map(axisEntries.map(entry => [entry.col, entry]));

                const fallbackPlus = data?.lambda_plus ?? null;
                const fallbackMinus = data?.lambda_minus ?? null;
                let lambdaInfo = {
                    lambda_plus: fallbackPlus,
                    lambda_minus: fallbackMinus
                };
                if (lambdaInfo.lambda_plus == null && lambdaInfo.lambda_minus == null) {
                    lambdaInfo = await ensureLambdaValues(
                        runId,
                        epoch,
                        treeId,
                        treeMaxIter,
                        API.getFComponentPerAxis,
                        fallbackPlus,
                        fallbackMinus
                    );
                    if (!renderSeqTracker.isCurrent(seq)) return;
                }
                data.lambda_plus = lambdaInfo.lambda_plus;
                data.lambda_minus = lambdaInfo.lambda_minus;

                const useLambdaGradient = lambdaGradientToggle ? lambdaGradientToggle.checked : false;
                let colorScalePlus = null;
                let colorScaleMinus = null;

                if (useLambdaGradient) {
                    const lambdaPlus = lambdaInfo.lambda_plus;
                    const lambdaMinus = lambdaInfo.lambda_minus;
                    const scales = UI.createLambdaColorScales(
                        lambdaPlus != null && isFinite(lambdaPlus) ? [lambdaPlus] : [],
                        lambdaMinus != null && isFinite(lambdaMinus) ? [lambdaMinus] : []
                    );
                    colorScalePlus = scales.colorScalePlus;
                    colorScaleMinus = scales.colorScaleMinus;
                }

                const useLogScale = logScaleToggle ? logScaleToggle.checked : true;
                data.axes.forEach(axisData => {
                    const entry = axisMap.get(axisData.col);
                    if (!entry) return;
                    const isSelectedTree = (candidatesSet && candidatesSet.has(Number(treeId)));
                    const isBestTree = (bestId !== null && bestId === Number(treeId));
                    renderSingleTreeAxisChart(entry.chartContainer, axisData, useLogScale, colorScalePlus, colorScaleMinus, data, isSelectedTree, isBestTree);
                });
            }
        } catch (err) {
            console.error('Failed to render:', err);
            if (renderSeqTracker.isCurrent(seq)) {
                showStatus(`Error: ${err.message}`);
            }
        }
    }

})();
