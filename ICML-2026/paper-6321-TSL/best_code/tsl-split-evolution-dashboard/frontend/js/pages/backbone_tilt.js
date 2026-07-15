(async function () {
    const runSelect = document.getElementById('runSelect');
    const epochSelect = document.getElementById('epochSelect');
    const treeSelect = document.getElementById('treeSelect');
    const iterationRange = document.getElementById('iterationRange');
    const iterationValue = document.getElementById('iterationValue');
    const axesGrid = document.getElementById('axesGrid');
    const mergedToggle = document.getElementById('mergedToggle');
    const combinedToggle = document.getElementById('combinedToggle');
    const logScaleToggle = document.getElementById('logScaleToggle');
    const lambdaGradientToggle = document.getElementById('lambdaGradientToggle');
    const highlightChoiceToggle = document.getElementById('highlightChoiceToggle');

    let currentRunId = null;
    let epochsTrees = {};
    const timelineCache = new Map();
    const combinedCache = new Map();
    const layoutManager = ComponentPageShared.createLayoutManager(axesGrid);
    const renderScheduler = ComponentPageShared.createRenderScheduler(() => render());
    const renderSeqTracker = ComponentPageShared.createRenderSequence();
    const lambdaCache = ComponentPageShared.createLambdaCache();

    const showStatus = (message) => ComponentPageShared.showStatus(axesGrid, layoutManager, message);
    const getSelectedTrees = () => ComponentPageShared.getSelectedTrees(treeSelect);

    function ensureLayout(layoutKey, columnsMeta) {
        return layoutManager.ensure({
            layoutKey,
            modeKey: 'backbone-tilt',
            columnsMeta,
            setupGrid(gridEl) {
                gridEl.style.display = 'grid';
                gridEl.style.gridTemplateColumns = '1fr 1fr';
                gridEl.style.gap = '20px';
            },
            createEntry(col, _mode, ctx) {
                const { axesGrid, createChartContainer } = ctx;
                const axisLabel = document.createElement('div');
                axisLabel.style.gridColumn = '1 / -1';
                axisLabel.style.textAlign = 'center';
                axisLabel.style.fontWeight = 'bold';
                axisLabel.style.padding = '10px';
                axisLabel.style.backgroundColor = '#f3f4f6';
                axisLabel.style.borderRadius = '4px';
                axisLabel.textContent = `Axis ${col}`;
                axesGrid.appendChild(axisLabel);

                const backboneContainer = createChartContainer();
                axesGrid.appendChild(backboneContainer);
                const tiltContainer = createChartContainer();
                axesGrid.appendChild(tiltContainer);

                return {
                    col,
                    labelEl: axisLabel,
                    backboneContainer,
                    tiltContainer
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

    async function ensureLambdaValues(runId, epoch, treeId, treeMaxIter, fallbackPlus = null, fallbackMinus = null) {
        return lambdaCache.ensure(
            runId,
            epoch,
            treeId,
            treeMaxIter,
            API.getComponentDecomposition,
            fallbackPlus,
            fallbackMinus
        );
    }

    async function fetchCombinedSnapshot(runId, epoch) {
        if (!Number.isFinite(epoch)) return null;
        const cacheKey = String(epoch);
        let promise = combinedCache.get(cacheKey);
        if (!promise) {
            promise = API.getCombinedForEpoch(epoch)
                .then(rows => (Array.isArray(rows) ? rows : []))
                .catch(err => {
                    combinedCache.delete(cacheKey);
                    throw err;
                });
            combinedCache.set(cacheKey, promise);
        }

        try {
            const entries = await promise;
            const runEntry = entries.find(item => Number(item?.run_id) === runId && item?.snapshot);
            if (!runEntry) return null;
            const snapshot = runEntry.snapshot || {};
            const hasBackbone = Array.isArray(snapshot.backbone_values) && snapshot.backbone_values.length > 0;
            const hasTilt = Array.isArray(snapshot.tilt_values) && snapshot.tilt_values.length > 0;
            if (!hasBackbone || !hasTilt) {
                console.warn('Combined grid lacks backbone/tilt values for run', runId, 'epoch', epoch);
                return null;
            }
            return runEntry;
        } catch (err) {
            console.error('Failed to fetch combined grid data:', err);
            return null;
        }
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
                // Try to pick the first run that has epochs/trees available
                let picked = false;
                for (const r of runs) {
                    runSelect.value = r.run_id;
                    try {
                        const et = await API.getEpochsTrees(r.run_id);
                        if (et && Object.keys(et).length > 0) {
                            picked = true;
                            break;
                        }
                    } catch (e) {
                        // ignore and try next
                    }
                }
                // Fallback to the first run if none had epochs
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
            epochsTrees = await API.getEpochsTrees(runId);
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

            // Extract tree ID if this is merged mode (key format: backbone_tree_X or tilt_tree_X)
            let treeId = null;
            let highlightPattern = key;
            if (key.includes('_tree_')) {
                treeId = key.split('_tree_')[1];
                // For merged mode, highlight all traces with this tree ID across all axes
                // Pattern: backbone_tree_X or tilt_tree_X
                highlightPattern = key; // Use the full key to match both backbone and tilt for this tree
            }

            // Highlight traces across all charts in axesGrid
            if (treeId) {
                // Merged mode: highlight this tree across all axes (both backbone and tilt)
                d3.selectAll('#axesGrid .chart svg .trace')
                    .attr('opacity', function () {
                        const traceKey = this.getAttribute('data-key');
                        // Match if it's the same tree (backbone or tilt for this tree)
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
            const baseLabel = comp.seriesLabel || (key.includes('tilt') ? 'Tilt' : 'Backbone');
            let titleLabel = baseLabel;
            if (key.includes('_tree_')) {
                titleLabel = `${baseLabel} (product ${treeId})`;
            }
            const fallbackColor = baseLabel === 'Tilt' ? '#ef4444' : '#3b82f6';
            const color = comp.color || fallbackColor;

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

    // Overlay combined backbone or tilt as dashed line
    function overlayCombinedComponent(containerEl, snapshot, dim, bounds, isBackbone, baseComps = [], useLogScale = null) {
        try {
            const snapshotData = snapshot.snapshot;
            const vals = isBackbone
                ? ((snapshotData.backbone_values || [])[dim] || [])
                : ((snapshotData.tilt_values || [])[dim] || []);
            const ilist = (snapshotData.intervals || [])[dim] || [];

            if (!ilist.length || !vals.length) {
                // Silently return - data might not be available for this dimension
                return;
            }

            const intervals = ilist.map((ib, i) => [UI.toFloat(ib[0]), UI.toFloat(ib[1]), UI.toFloat(vals[i])]).filter(v => v.length === 3);
            const { innerW, innerH } = Charts.sizeOf(containerEl);
            const svg = d3.select(containerEl).select('svg');

            // Reuse same scales as main chart
            const [lo, hi] = [bounds.min, bounds.max];
            const x = d3.scaleLinear().domain([lo, hi]).range([0, innerW]);

            // Compute Y scale from base components to match the existing chart
            // Use provided log scale, or default to backbone=log, tilt=linear
            const useLog = useLogScale !== null ? useLogScale : (isBackbone ? true : false);
            function sy(v) { return Math.sign(v) * Math.log10(1 + Math.abs(v)); }
            const eps = 1e-12;

            let ymin = Infinity, ymax = -Infinity;
            if (Array.isArray(baseComps) && baseComps.length > 0) {
                baseComps.forEach(bc => {
                    const ints = bc.intervals || [];
                    const allZero = ints.every(iv => Math.abs(iv[2] || 0) < eps);
                    const baseVal = useLog ? sy(1) : 1;
                    ints.forEach(iv => {
                        const v = iv[2] == null ? 0 : +iv[2];
                        const tv = allZero ? baseVal : (useLog ? sy(v) : v);
                        ymin = Math.min(ymin, tv);
                        ymax = Math.max(ymax, tv);
                    });
                });
            }

            // Also include combined values in bounds
            intervals.forEach(iv => {
                const v = iv[2] == null ? 0 : +iv[2];
                const tv = useLog ? sy(v) : v;
                ymin = Math.min(ymin, tv);
                ymax = Math.max(ymax, tv);
            });

            if (!isFinite(ymin) || !isFinite(ymax)) { ymin = -1; ymax = 1; }
            if (ymin === ymax) { ymin -= 1; ymax += 1; }
            const y = d3.scaleLinear().domain([ymin, ymax]).nice().range([innerH, 0]);

            const points = [];
            intervals.forEach(([a, b, v], i) => {
                const left = Number.isFinite(a) ? a : lo;
                const right = Number.isFinite(b) ? b : hi;
                const vy = useLog ? sy(v) : v;
                points.push([left, vy], [right, vy]);
            });

            if (points.length >= 2) {
                const line = d3.line().x(d => x(d[0])).y(d => y(d[1])).curve(d3.curveStepAfter);
                const g = svg.select('g');
                if (g.node()) {
                    g.append('path')
                        .datum(points)
                        .attr('fill', 'none')
                        .attr('stroke', '#000')  // Black for combined
                        .attr('stroke-width', 1.4 * 1.5)
                        .attr('stroke-dasharray', '4,3')
                        .attr('opacity', 0.9)
                        .attr('d', line)
                        .attr('class', 'trace combined-overlay')
                        .attr('data-key', `combined_${isBackbone ? 'backbone' : 'tilt'}`);
                } else {
                    console.warn('SVG group element not found');
                }
            } else {
                console.warn('Not enough points to draw overlay:', points.length);
            }
        } catch (e) {
            console.warn('Failed to overlay combined component:', e);
        }
    }

    function renderAxisChart(container, axisData, isBackbone, combinedSnapshot = null, useLogScale = null, lambdaPlus = null, lambdaMinus = null, colorScale = null, isSelected = false, isBest = false) {
        if (typeof Charts === 'undefined') {
            container.innerHTML = '<p>Error: Charts module not loaded. Please refresh the page.</p>';
            console.error('Charts is not defined. Make sure charts.js is loaded before backbone_tilt.js');
            return;
        }

        // Ensure container has proper sizing
        if (container.style.width === '' || container.style.width === 'auto') {
            const rect = container.getBoundingClientRect();
            if (rect.width > 0) {
                container.style.width = `${rect.width}px`;
            }
        }

        const intervals = isBackbone ? axisData.intervals_backbone : axisData.intervals_tilt;
        const title = isBackbone ? `Backbone (Axis ${axisData.col})` : `Tilt (Axis ${axisData.col})`;

        if (!intervals || intervals.length === 0) {
            container.innerHTML = '<p>No data</p>';
            return;
        }

        const bounds = UI.computeBoundsFromIntervalsTriples(intervals);

        // Determine color based on toggle
        const useLambdaGradient = lambdaGradientToggle ? lambdaGradientToggle.checked : false;
        let lineColor = isBackbone ? '#3b82f6' : '#ef4444'; // Default colors
        if (useLambdaGradient && colorScale && lambdaPlus != null && lambdaMinus != null) {
            const lambdaSum = Math.abs(lambdaPlus) + Math.abs(lambdaMinus);
            if (isFinite(lambdaSum)) {
                lineColor = colorScale(lambdaSum);
            }
        }
        // Best takes precedence (yellow), otherwise selected candidates are green
        if (isBest) {
            lineColor = '#f59e0b';
        } else if (isSelected) {
            lineColor = '#10b981';
        }

        // Format intervals for Charts.stepFunctions: [[a, b, val], ...]
        const lambdaInfo = {
            lambda_plus: lambdaPlus,
            lambda_minus: lambdaMinus
        };
        const comps = [{
            intervals: intervals,
            key: isBackbone ? 'backbone' : 'tilt',
            color: lineColor,
            seriesLabel: isBackbone ? 'Backbone' : 'Tilt',
            lambdaInfo
        }];

        // Double-check Charts is available before calling
        if (typeof Charts === 'undefined' || typeof Charts.stepFunctions !== 'function') {
            container.innerHTML = '<p>Error: Charts library not loaded. Please refresh the page.</p>';
            console.error('Charts.stepFunctions is not available');
            return;
        }

        // Determine log scale: use toggle if provided, otherwise default to backbone=log, tilt=linear
        const useLog = useLogScale !== null ? useLogScale : (isBackbone ? true : false);

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

            // Overlay combined component if available
            if (combinedSnapshot) {
                overlayCombinedComponent(container, combinedSnapshot, axisData.col, bounds, isBackbone, comps, useLog);
            }
        } catch (err) {
            container.innerHTML = `<p>Error rendering chart: ${err.message}</p>`;
            console.error('Error in Charts.stepFunctions:', err);
        }
    }

    function renderMergedChart(container, allTreesData, isBackbone, axisCol, combinedSnapshot = null, useLogScale = null, lambdaDataMap = null, colorScale = null, candidatesSet = null, bestId = null) {
        if (typeof Charts === 'undefined') {
            container.innerHTML = '<p>Error: Charts module not loaded. Please refresh the page.</p>';
            console.error('Charts is not defined. Make sure charts.js is loaded before backbone_tilt.js');
            return;
        }

        // Ensure container has proper sizing
        if (container.style.width === '' || container.style.width === 'auto') {
            const rect = container.getBoundingClientRect();
            if (rect.width > 0) {
                container.style.width = `${rect.width}px`;
            }
        }

        // allTreesData is an array of { treeId, intervals_backbone, intervals_tilt }
        if (!allTreesData || allTreesData.length === 0) {
            container.innerHTML = '<p>No data</p>';
            return;
        }

        // Get the appropriate intervals (backbone or tilt) from all trees
        const intervalsToUse = isBackbone ? 'intervals_backbone' : 'intervals_tilt';

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

            // Use gradient color if enabled
            if (useLambdaGradient && lambdaData && colorScale) {
                const lambdaPlus = lambdaData.lambda_plus;
                const lambdaMinus = lambdaData.lambda_minus;
                if (lambdaPlus != null && lambdaMinus != null) {
                    const lambdaSum = Math.abs(lambdaPlus) + Math.abs(lambdaMinus);
                    if (isFinite(lambdaSum)) {
                        treeColor = colorScale(lambdaSum);
                        console.debug(`Tree ${treeId} lambdaSum=${lambdaSum}, color=${treeColor}`);
                    }
                }
            }
            // If this tree is part of the combination choice, color accordingly:
            // best -> yellow, candidates -> green
            try {
                const tidNum = Number(treeId);
                if (bestId !== null && bestId === tidNum) {
                    treeColor = '#f59e0b';
                } else if (candidatesSet && candidatesSet.has(tidNum)) {
                    treeColor = '#10b981';
                }
            } catch (e) {
                // ignore parsing errors
            }

            const intervals = treeData[intervalsToUse] || [];

            if (intervals.length > 0) {
                comps.push({
                    intervals: intervals,
                    key: `${isBackbone ? 'backbone' : 'tilt'}_tree_${treeId}`,
                    color: treeColor,
                    split_label: `T${treeId}`,
                    seriesLabel: isBackbone ? 'Backbone' : 'Tilt',
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

        // Determine log scale: use toggle if provided, otherwise default to backbone=log, tilt=linear
        const useLog = useLogScale !== null ? useLogScale : (isBackbone ? true : false);

        try {
            const title = isBackbone
                ? `Backbone (Axis ${axisCol}, Merged)`
                : `Tilt (Axis ${axisCol}, Merged)`;
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

            // Overlay combined component if available
            if (combinedSnapshot) {
                overlayCombinedComponent(container, combinedSnapshot, axisCol, bounds, isBackbone, comps, useLog);
            }
        } catch (err) {
            container.innerHTML = `<p>Error rendering chart: ${err.message}</p>`;
            console.error('Error in Charts.stepFunctions:', err);
        }
    }

    async function render() {
        const seq = renderSeqTracker.next();
        const runId = parseInt(runSelect.value);
        const epoch = parseInt(epochSelect.value);
        const iteration = parseInt(iterationRange.value);
        const selectedTrees = getSelectedTrees();
        const merged = !!mergedToggle?.checked;
        const combined = !!combinedToggle?.checked;
        const selectionKey = selectedTrees.slice().sort((a, b) => a - b).join(',') || 'all';
        const layoutKey = merged
            ? `merged:${runId}:${epoch}:${selectionKey}:${combined ? 1 : 0}`
            : `single:${runId}:${epoch}:${selectionKey}:${combined ? 1 : 0}`;

        if (![runId, epoch, iteration].every(Number.isFinite)) {
            showStatus('Please select run, epoch, and iteration');
            return;
        }
        if (!merged && selectedTrees.length !== 1) {
            showStatus('Please select exactly one tree when merged is off');
            return;
        }

        if (!layoutManager.hasLayout()) {
            axesGrid.innerHTML = '<p>Loading...</p>';
            axesGrid.style.display = 'block';
        }

        let combinedSnapshot = null;
        if (combined) {
            combinedSnapshot = await fetchCombinedSnapshot(runId, epoch);
        }
        // Attempt to fetch combination choice (optional). Used to highlight selected trees.
        // Prepare combination choice info only if highlight toggle is enabled
        let comboChoice = null;
        let candidatesSet = new Set();
        let bestId = null;
        if (highlightChoiceToggle && highlightChoiceToggle.checked) {
            try {
                comboChoice = await API.getCombinationChoice(runId, epoch);
            } catch (e) {
                comboChoice = null;
            }
            candidatesSet = new Set((comboChoice && comboChoice.candidates) ? comboChoice.candidates.map(c => Number(c.tree_id)) : []);
            bestId = comboChoice && Number.isFinite(+comboChoice.best_index) ? +comboChoice.best_index : null;
        }

        try {
            if (merged) {
                const epochKey = String(epoch);
                const epochTrees = (epochsTrees[epochKey] || []).sort((a, b) => a - b);

                if (epochTrees.length === 0) {
                    showStatus('No trees available for this epoch');
                    return;
                }

                const targetTrees = selectedTrees.length
                    ? epochTrees.filter(t => selectedTrees.includes(t))
                    : epochTrees;
                if (targetTrees.length === 0) {
                    showStatus('Selected trees are not available for this epoch');
                    return;
                }

                const tl = await getRunTimeline(runId);
                if (!renderSeqTracker.isCurrent(seq)) return;
                const treeMaxIters = calculateTreeMaxIters(tl, epoch, targetTrees);

                let allTreesData = null;
                try {
                    const multiKey = UI.Cache.makeKey('component_decomposition_multi', runId, epoch, iteration, targetTrees.join(','));
                    const multi = await UI.Cache.fetchOrGet(
                        multiKey,
                        () => API.getComponentDecompositionMulti(runId, epoch, iteration, targetTrees)
                    );
                    if (!renderSeqTracker.isCurrent(seq)) return;
                    if (multi && Array.isArray(multi.trees)) {
                        allTreesData = multi.trees.map(t => {
                            const treeId = Number(t.tree_id);
                            const treeMaxIter = treeMaxIters.get(treeId) ?? Number(t.iter_no) ?? iteration;
                            return {
                                treeId,
                                data: {
                                    components: t.components || [],
                                    lambda_plus: t.lambda_plus ?? null,
                                    lambda_minus: t.lambda_minus ?? null
                                },
                                treeMaxIter
                            };
                        });
                    }
                } catch (e) {
                    console.warn('Multi-tree backbone/tilt fetch failed; falling back to per-tree requests:', e);
                }

                if (!allTreesData) {
                    allTreesData = await Promise.all(
                        targetTrees.map(async treeId => {
                            const treeMaxIter = treeMaxIters.get(treeId) || 0;
                            const iterToUse = Math.min(iteration, treeMaxIter);
                            try {
                                const cacheKey = UI.Cache.makeKey('component_decomposition', runId, epoch, treeId, iterToUse);
                                const data = await UI.Cache.fetchOrGet(cacheKey, () =>
                                    API.getComponentDecomposition(runId, epoch, treeId, iterToUse)
                                );
                                return { treeId, data, treeMaxIter };
                            } catch (err) {
                                console.warn(`Failed to fetch data for tree ${treeId} at iter ${iterToUse}:`, err);
                                return { treeId, data: null, treeMaxIter };
                            }
                        })
                    );
                    if (!renderSeqTracker.isCurrent(seq)) return;
                }

                const firstTreeWithData = allTreesData.find(t => t.data && Array.isArray(t.data.components) && t.data.components.length > 0);
                if (!firstTreeWithData) {
                    showStatus('No data available for selected trees');
                    return;
                }

                const columnsMeta = firstTreeWithData.data.components.map(c => c.col);
                const axisEntries = ensureLayout(layoutKey, columnsMeta);
                const axisMap = new Map(axisEntries.map(entry => [entry.col, entry]));

                const useLambdaGradient = lambdaGradientToggle ? lambdaGradientToggle.checked : false;
                const lambdaDataMap = new Map();
                let lambdaSumValues = [];
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
                    if (useLambdaGradient) {
                        const lambdaSum = Math.abs(vals.lambda_plus || 0) + Math.abs(vals.lambda_minus || 0);
                        if (isFinite(lambdaSum)) lambdaSumValues.push(lambdaSum);
                    }
                });

                const colorScale = useLambdaGradient && lambdaSumValues.length > 0
                    ? UI.createLambdaColorScales(lambdaSumValues, [], {
                        interpolatePlus: d3.interpolateInferno
                    }).colorScalePlus
                    : null;

                for (let axisIdx = 0; axisIdx < columnsMeta.length; axisIdx++) {
                    if (!renderSeqTracker.isCurrent(seq)) return;
                    const col = columnsMeta[axisIdx];
                    const entry = axisMap.get(col);
                    if (!entry) continue;

                    const allTreesAxisData = [];
                    allTreesData.forEach(({ treeId, data }) => {
                        if (!data || !Array.isArray(data.components)) return;
                        const component = data.components.find(c => c.col === col) || data.components[axisIdx];
                        if (!component) return;

                        const intervals = component.intervals || [];
                        const backbone = component.backbone || [];
                        const tilt = component.tilt || [];
                        const intervalsBackbone = [];
                        const intervalsTilt = [];

                        for (let i = 0; i < intervals.length && i < backbone.length && i < tilt.length; i++) {
                            const interval = intervals[i];
                            if (!interval || interval.length < 2) continue;
                            intervalsBackbone.push([interval[0], interval[1], backbone[i]]);
                            intervalsTilt.push([interval[0], interval[1], tilt[i]]);
                        }

                        if (intervalsBackbone.length === 0 && intervalsTilt.length === 0) return;

                        allTreesAxisData.push({
                            treeId,
                            intervals_backbone: intervalsBackbone,
                            intervals_tilt: intervalsTilt
                        });
                    });

                    if (allTreesAxisData.length === 0) {
                        entry.backboneContainer.innerHTML = '<p>No data</p>';
                        entry.tiltContainer.innerHTML = '<p>No data</p>';
                        continue;
                    }

                    const useLogScale = logScaleToggle ? logScaleToggle.checked : null;
                    renderMergedChart(entry.backboneContainer, allTreesAxisData, true, col, combinedSnapshot, useLogScale, lambdaDataMap, colorScale, candidatesSet, bestId);
                    renderMergedChart(entry.tiltContainer, allTreesAxisData, false, col, combinedSnapshot, useLogScale, lambdaDataMap, colorScale, candidatesSet, bestId);
                }
            } else {
                const treeId = selectedTrees[0];
                const tl = await getRunTimeline(runId);
                if (!renderSeqTracker.isCurrent(seq)) return;
                const treeMaxIters = calculateTreeMaxIters(tl, epoch, [treeId]);
                const treeMaxIter = treeMaxIters.get(treeId) || 0;
                const iterToUse = Math.min(iteration, treeMaxIter);
                const cacheKey = UI.Cache.makeKey('component_decomposition', runId, epoch, treeId, iterToUse);
                const data = await UI.Cache.fetchOrGet(cacheKey, () =>
                    API.getComponentDecomposition(runId, epoch, treeId, iterToUse)
                );
                if (!renderSeqTracker.isCurrent(seq)) return;

                if (!data.components || data.components.length === 0) {
                    showStatus('No data available');
                    return;
                }

                const columnsMeta = data.components.map(c => c.col);
                const axisEntries = ensureLayout(layoutKey, columnsMeta);
                const axisMap = new Map(axisEntries.map(entry => [entry.col, entry]));

                let lambdaInfo = {
                    lambda_plus: data?.lambda_plus ?? null,
                    lambda_minus: data?.lambda_minus ?? null
                };
                if (lambdaInfo.lambda_plus == null && lambdaInfo.lambda_minus == null) {
                    lambdaInfo = await ensureLambdaValues(
                        runId,
                        epoch,
                        treeId,
                        treeMaxIter,
                        data?.lambda_plus ?? null,
                        data?.lambda_minus ?? null
                    );
                    if (!renderSeqTracker.isCurrent(seq)) return;
                }
                data.lambda_plus = lambdaInfo.lambda_plus;
                data.lambda_minus = lambdaInfo.lambda_minus;

                const useLambdaGradient = lambdaGradientToggle ? lambdaGradientToggle.checked : false;
                let colorScale = null;
                if (useLambdaGradient) {
                    const lambdaSum = Math.abs(lambdaInfo.lambda_plus || 0) + Math.abs(lambdaInfo.lambda_minus || 0);
                    if (isFinite(lambdaSum)) {
                        const scales = UI.createLambdaColorScales([lambdaSum], [], {
                            interpolatePlus: d3.interpolateInferno
                        });
                        colorScale = scales.colorScalePlus;
                    }
                }

                data.components.forEach(component => {
                    const entry = axisMap.get(component.col);
                    if (!entry) return;

                    const intervals = component.intervals || [];
                    const intervalsBackbone = [];
                    const intervalsTilt = [];

                    const backbone = component.backbone || [];
                    const tilt = component.tilt || [];
                    for (let i = 0; i < intervals.length && i < backbone.length && i < tilt.length; i++) {
                        const interval = intervals[i];
                        if (!interval || interval.length < 2) continue;
                        intervalsBackbone.push([interval[0], interval[1], backbone[i]]);
                        intervalsTilt.push([interval[0], interval[1], tilt[i]]);
                    }

                    const axisData = {
                        col: component.col,
                        intervals_backbone: intervalsBackbone,
                        intervals_tilt: intervalsTilt
                    };

                    const useLogScale = logScaleToggle ? logScaleToggle.checked : null;
                    const isSelectedTree = (candidatesSet && candidatesSet.has(Number(treeId)));
                    const isBestTree = (bestId !== null && bestId === Number(treeId));
                    renderAxisChart(entry.backboneContainer, axisData, true, combinedSnapshot, useLogScale, lambdaInfo.lambda_plus, lambdaInfo.lambda_minus, colorScale, isSelectedTree, isBestTree);
                    renderAxisChart(entry.tiltContainer, axisData, false, combinedSnapshot, useLogScale, lambdaInfo.lambda_plus, lambdaInfo.lambda_minus, colorScale, isSelectedTree, isBestTree);
                });
            }
        } catch (err) {
            console.error('Failed to render:', err);
            if (renderSeqTracker.isCurrent(seq)) {
                showStatus(`Error: ${err.message}`);
            }
        }
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

    combinedToggle.addEventListener('change', async () => {
        // Combined toggle doesn't affect tree selector - it's just an overlay
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
                await updateIterationBounds();
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

    // Initialize
    try {
        await waitForCharts();
        await loadRuns();
    } catch (err) {
        console.error('Failed to initialize:', err);
        showStatus(`Error: ${err.message}`);
    }
})();
