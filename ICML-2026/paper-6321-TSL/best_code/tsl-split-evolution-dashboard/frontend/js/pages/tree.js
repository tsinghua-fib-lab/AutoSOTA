(async function () {
    const runSelect = document.getElementById('runSelect');
    const epochSelect = document.getElementById('epochSelect');
    const treeSelect = document.getElementById('treeSelect');
    const iterationRange = document.getElementById('iterationRange');
    const iterationValue = document.getElementById('iterationValue');
    const mergedToggle = document.getElementById('mergedToggle');
    const iterationSlider = document.getElementById('iterationRange');
    const combinedOverlayToggle = document.getElementById('combinedOverlayToggle');

    // ===================== UTILITY FUNCTIONS =====================

    /**
     * Updates the event metrics display with split event data
     */
    function updateEventMetrics(splitEvent, showIteration = false) {
        if (!splitEvent) {
            document.getElementById('eventMetrics').style.display = 'none';
            return;
        }

        const col = (typeof splitEvent.col === 'number') ? splitEvent.col : Number(splitEvent.col || 0);
        const sv = (splitEvent.split_value != null) ? Number(splitEvent.split_value) : null;
        const gain = (splitEvent.gain != null) ? Number(splitEvent.gain) : null;
        const ua = (splitEvent.update_a != null) ? Number(splitEvent.update_a) : null;
        const ub = (splitEvent.update_b != null) ? Number(splitEvent.update_b) : null;
        const iterPart = showIteration && (splitEvent.iter_no != null) ? Number(splitEvent.iter_no) : null;

        // Handle special case for multiple trees
        let actionText = String(splitEvent.action || '').toLowerCase();
        let colText = col;

        if (splitEvent.action === 'multiple') {
            actionText = 'multiple trees';
            colText = `${col} trees`;
        }

        // Update the metric containers
        document.getElementById('actionValue').textContent = actionText;
        document.getElementById('colValue').textContent = colText;
        document.getElementById('splitValue').textContent = sv != null ? sv.toPrecision(6) : '—';
        document.getElementById('gainValue').textContent = gain != null ? gain.toPrecision(6) : '—';
        document.getElementById('leftValue').textContent = ua != null ? ua.toPrecision(6) : '—';
        document.getElementById('rightValue').textContent = ub != null ? ub.toPrecision(6) : '—';
        document.getElementById('iterValue').textContent = showIteration && iterPart != null ? iterPart : '—';

        // Update error metric if available
        const currentError = splitEvent.current_error;
        document.getElementById('errorValue').textContent = currentError != null ? currentError.toPrecision(6) : '—';

        // Show the metrics container
        document.getElementById('eventMetrics').style.display = 'flex';
    }

    /**
     * Finds the latest split event from a collection of trees
     * If multiple trees are selected, shows average gain across all trees
     */
    function findLatestSplitEvent(trees) {
        if (!trees || trees.length === 0) return null;

        // If only one tree, return its split event
        if (trees.length === 1) {
            return trees[0].split_event || null;
        }

        // For multiple trees, collect all split events and calculate averages
        const events = [];
        for (const t of trees) {
            if (t.split_event) {
                // Copy the split event and add current_error if available
                const eventWithError = { ...t.split_event };
                if (t.current_error !== undefined) {
                    eventWithError.current_error = t.current_error;
                }
                events.push(eventWithError);
            }
        }

        if (events.length === 0) return null;
        if (events.length === 1) return events[0];

        // Calculate average gain and current error across all trees
        const totalGain = events.reduce((sum, ev) => sum + (ev.gain || 0), 0);
        const avgGain = totalGain / events.length;

        // Calculate average current error (use err_after if available, otherwise current_error)
        const totalError = events.reduce((sum, ev) => {
            const error = ev.current_error || ev.err_after;
            return sum + (error || 0);
        }, 0);
        const avgError = totalError / events.length;

        // Return a summary event with average values
        return {
            action: 'multiple',
            col: events.length,
            split_value: null,
            gain: avgGain,
            update_a: null,
            update_b: null,
            iter_no: events[0].iter_no, // All should be the same iteration
            current_error: avgError
        };
    }

    /**
     * Sets up grid layout and checks if rebuild is needed
     */
    function setupGridLayout(container, nDims, lastLayout, runId, epoch, merged) {
        const cols = Math.max(1, Math.ceil(Math.sqrt(nDims)));
        const needsRebuild = (
            lastLayout.merged !== merged ||
            lastLayout.runId !== runId ||
            lastLayout.epoch !== epoch ||
            lastLayout.nDims !== nDims ||
            lastLayout.cols !== cols
        );

        if (needsRebuild) {
            container.selectAll('*').remove();
            container.style('grid-template-columns', `repeat(${cols}, minmax(300px, 1fr))`);
        }

        return { needsRebuild, cols };
    }

    /**
     * Creates hover interactions for charts
     */
    function createHoverInteractions(svg, tooltip, bounds, comps, treeColors, combinedSnapshot, dim, useLog) {
        svg.on('mousemove', function (event) {
            const target = event.target;
            if (!target || target.tagName !== 'path') return;
            const key = target.getAttribute('data-key');
            if (!key) return;

            // Highlight current trace
            d3.selectAll('#productList .chart svg .trace')
                .attr('opacity', function () {
                    return this.getAttribute('data-key') === key ? 1.0 : 0.15;
                })
                .attr('stroke-width', function () {
                    return this.getAttribute('data-key') === key ? 2.0 : 1.0;
                });

            // Determine title and color
            let titleLabel = '';
            let color = treeColors[key] || '#444';
            if (key === 'combined_overlay') {
                titleLabel = 'Combined';
                color = '#111';
            } else {
                titleLabel = `Product ${key.replace('tree_', '')}`;
            }

            // Compute value at cursor
            const { innerW } = Charts.sizeOf(svg.node().parentElement);
            const [mx, my] = d3.pointer(event, svg.select('g').node());
            const x = d3.scaleLinear().domain([bounds.min, bounds.max]).range([0, innerW]);
            const xVal = x.invert(mx);
            let valueAtCursor = null;

            if (key === 'combined_overlay' && combinedSnapshot) {
                valueAtCursor = getValueFromCombinedSnapshot(combinedSnapshot, dim, xVal, bounds);
            } else {
                valueAtCursor = getValueFromComponent(comps, key, xVal, bounds, dim);
            }

            // Show tooltip
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 12) + 'px';
            tooltip.style.top = (event.pageY + 12) + 'px';
            const valStr = (valueAtCursor != null && isFinite(+valueAtCursor)) ? Number(valueAtCursor).toPrecision(6) : '—';

            if (comps.length > 1) {
                tooltip.innerHTML = `<div style="display:flex;align-items:center;gap:8px"><span style="display:inline-block;width:10px;height:10px;background:${color};border-radius:2px"></span><b>${titleLabel}</b></div><div>Component ${dim}</div><div>value: ${valStr}</div>`;
            } else {
                tooltip.innerHTML = `<b>Component ${dim}</b><div>value: ${valStr}</div>`;
            }
        });

        svg.on('mouseleave', function () {
            if (comps.length > 1) {
                d3.selectAll('#productList .chart svg .trace')
                    .attr('opacity', 1.0)
                    .attr('stroke-width', 1.4);
            }
            const t = Charts.getTooltip();
            if (t) t.style.display = 'none';
        });
    }

    /**
     * Gets value from primary snapshot at cursor position
     */
    function getValueFromCombinedSnapshot(combinedSnapshot, dim, xVal, bounds) {
        const vals = (combinedSnapshot.snapshot.grid_values || [])[dim] || [];
        const ilist = (combinedSnapshot.snapshot.intervals || [])[dim] || [];
        const intervals = ilist.map((ib, i) => [UI.toFloat(ib[0]), UI.toFloat(ib[1]), UI.toFloat(vals[i])]).filter(v => v.length === 3);

        for (const [a, b, v] of intervals) {
            const left = Number.isFinite(a) ? a : bounds.min;
            const right = Number.isFinite(b) ? b : bounds.max;
            if (xVal >= left && xVal < right) return v;
        }

        if (intervals.length > 0) {
            return intervals[intervals.length - 1][2];
        }
        return null;
    }

    /**
     * Gets value from component at cursor position
     */
    function getValueFromComponent(comps, key, xVal, bounds, dim) {
        const comp = Array.isArray(comps) ? comps.find(c => (c.key || `tree_${c.tree_id}`) === key) : comps[0];
        if (!comp || !Array.isArray(comp.intervals)) return null;

        for (const [a, b, v] of comp.intervals) {
            const left = Number.isFinite(a) ? a : bounds.min;
            const right = Number.isFinite(b) ? b : bounds.max;
            if (xVal >= left && xVal < right) return v;
        }

        if (comp.intervals.length > 0) {
            // For single tree mode, handle special case
            if (!Array.isArray(comps)) {
                return comp.intervals[comp.intervals.length - 1][2];
            }
            // Fallback to last interval value for multi-tree
            const last = comp.intervals[comp.intervals.length - 1];
            return last[2];
        }
        return null;
    }

    /**
     * Renders charts for given dimension with common setup
     */
    function renderChartForDimension(container, dim, nDims, data, bounds, identified, useLog, combinedSnapshot, treeColors, needsRebuild) {
        let el;
        if (needsRebuild) {
            el = document.createElement('div');
            el.className = 'chart';
            container.node().appendChild(el);
        } else {
            el = container.node().children[dim];
        }

        const actualBounds = bounds && (bounds[String(dim)] || bounds[dim]) || { min: -1, max: 1 };
        return { el, bounds: actualBounds };
    }

    // ===================== END UTILITY FUNCTIONS =====================

    await UI.loadRunsTo(runSelect);
    // Apply run from URL/localStorage if present
    const params = new URLSearchParams(window.location.search);
    const urlRun = params.get('run');
    const urlEpoch = params.get('epoch');
    const urlTrees = params.get('trees');
    let urlAppliedOnce = false; // ensure we only auto-apply URL tree selection once
    const storedRun = localStorage.getItem('selectedRunId');
    const preferred = [urlRun, storedRun].find(Boolean);
    if (preferred && Array.from(runSelect.options).some(o => o.value === String(preferred))) {
        runSelect.value = String(preferred);
    }

    const timelineCache = new Map();
    async function getRunTimeline(runId) {
        if (!timelineCache.has(runId)) {
            const tl = await API.getTimeline(runId);
            timelineCache.set(runId, Array.isArray(tl) ? tl : []);
        }
        return timelineCache.get(runId);
    }

    // Primary overlay: no UI disabling; overlay on top of current view

    async function updateIterationBounds() {
        const runId = +runSelect.value;
        const epoch = +epochSelect.value;
        const selectedTrees = Array.from(treeSelect.selectedOptions).map(opt => +opt.value);
        const merged = !!mergedToggle?.checked;

        console.log('updateIterationBounds called with:', { runId, epoch, selectedTrees, merged });

        if (![runId, epoch].every(Number.isFinite) || (!merged && selectedTrees.length === 0)) return;

        const tl = await getRunTimeline(runId);
        let relevant = tl.filter(e => Number(e.epoch) === epoch);
        if (!merged) {
            // For multi-select, use the first selected tree for iteration bounds
            const treeId = selectedTrees[0];
            relevant = relevant.filter(e => Number(e.tree_id) === treeId);
        }
        const maxIter = relevant.length ? Math.max(...relevant.map(e => Number(e.iter_no) || 0)) : 0;
        iterationSlider.max = String(Math.max(0, maxIter));
        if (Number(iterationSlider.value) > maxIter) {
            iterationSlider.value = String(maxIter);
            iterationValue.textContent = String(maxIter);
        }
    }

    async function loadEpochsTrees() {
        const runId = +runSelect.value;
        if (!Number.isFinite(runId)) return;
        const et = await API.getEpochsTrees(runId);
        const epochs = Object.keys(et).map(e => +e).sort((a, b) => a - b);
        UI.setOptions(epochSelect, epochs.map(e => ({ label: `Stage ${e}`, value: e })));

        // Apply URL parameters if present
        if (urlEpoch && epochs.includes(+urlEpoch)) {
            epochSelect.value = urlEpoch;
        } else if (epochs.length > 0) {
            epochSelect.value = epochs[0];
        }

        await refreshTreeList();
        await updateIterationBounds();
    }

    async function refreshTreeList() {
        const runId = +runSelect.value;
        const et = await API.getEpochsTrees(runId);
        const epoch = +epochSelect.value || +Object.keys(et)[0] || 0;
        const trees = (et[String(epoch)] || []).map(t => ({ label: `Product ${t}`, value: t }));
        UI.setOptions(treeSelect, trees);

        // Apply URL tree selection if present (once)
        if (urlTrees && trees.length > 0 && !urlAppliedOnce) {
            const treeIds = urlTrees.split(',').map(id => +id.trim());
            // Clear previous selection
            Array.from(treeSelect.options).forEach(opt => opt.selected = false);
            // Select trees from URL
            treeIds.forEach(treeId => {
                const option = Array.from(treeSelect.options).find(opt => +opt.value === treeId);
                if (option) {
                    option.selected = true;
                }
            });
            // If no trees were selected from URL, select first tree by default
            if (Array.from(treeSelect.selectedOptions).length === 0 && trees.length > 0) {
                treeSelect.selectedIndex = 0;
            }

            // Show selection info
            const selectionInfo = document.getElementById('selectionInfo');
            if (selectionInfo) {
                const selectedCount = Array.from(treeSelect.selectedOptions).length;
                selectionInfo.textContent = `Showing ${selectedCount} selected trees from error plot selection`;
                selectionInfo.style.display = 'block';
            }

            // Mark URL selection as applied
            urlAppliedOnce = true;
        } else if (trees.length > 0) {
            // Select first tree by default
            treeSelect.selectedIndex = 0;

            // Hide selection info
            const selectionInfo = document.getElementById('selectionInfo');
            if (selectionInfo) {
                selectionInfo.style.display = 'none';
                // Show event metrics when hiding selection info
                document.getElementById('eventMetrics').style.display = 'flex';
            }
        }

        iterationRange.value = '0';
        iterationValue.textContent = '0';
        await updateIterationBounds();
        render();
    }

    let lastLayout = { merged: null, runId: null, epoch: null, nDims: 0, cols: 0 };

    async function render() {
        const runId = +runSelect.value;
        const epoch = +epochSelect.value;
        const selectedTrees = Array.from(treeSelect.selectedOptions).map(opt => +opt.value);
        const iteration = +iterationRange.value;

        // Debug logging
        console.log('Render called with:', { runId, epoch, selectedTrees, iteration });

        const merged = !!mergedToggle?.checked;
        if (![runId, epoch].every(Number.isFinite) || (!merged && selectedTrees.length === 0)) return;
        const primary = !!combinedOverlayToggle?.checked;
        const identified = document.getElementById('identifiedToggle')?.checked === true;
        const useLog = document.getElementById('logToggle')?.checked === true;
        // If primary overlay requested, fetch snapshot for current run/epoch once
        let combinedSnapshot = null;
        if (primary) {
            try {
                const res = await API.getCombinedForEpoch(epoch);
                combinedSnapshot = Array.isArray(res) ? res.find(r => r && r.run_id === runId && r.snapshot && r.snapshot.grid_values) : null;
            } catch (e) {
                combinedSnapshot = null;
            }
        }
        const identified2 = identified; // keep usage below consistent
        let nDims = 0;
        const container = d3.select('#productList');
        let rebuildGrid = false;
        if (merged) {
            // "All:" mode - show all trees merged
            treeSelect.disabled = true;
            const identified = document.getElementById('identifiedToggle')?.checked === true;
            const selectedParam = (selectedTrees && selectedTrees.length > 0)
                ? selectedTrees.join(',')
                : '-1'; // fallback to all trees when none explicitly selected
            const data = await API.getUnifiedTreeComponents(
                runId,
                epoch,
                iteration,
                identified,
                selectedParam
            );
            nDims = data.n_dims || 0;
            if (!nDims) return;

            // Setup grid layout
            const { needsRebuild, cols } = setupGridLayout(container, nDims, lastLayout, runId, epoch, true);
            rebuildGrid = needsRebuild;
            if (needsRebuild) {
                lastLayout = { merged: true, runId, epoch, nDims, cols };
            }

            // Setup tree colors
            const palette = (Charts.__theme && Charts.__theme.palette) || [];
            const treeColors = {};
            (data.trees || []).forEach((t, i) => {
                treeColors[`tree_${t.tree_id}`] = palette[i % (palette.length || 7)] || '#555';
            });

            // Update event metrics
            const latestEvent = findLatestSplitEvent(data.trees);
            updateEventMetrics(latestEvent, false);

            // Render charts for each dimension
            for (let dim = 0; dim < nDims; dim++) {
                const { el, bounds } = renderChartForDimension(container, dim, nDims, data, data.bounds, identified, useLog, combinedSnapshot, treeColors, rebuildGrid);

                const comps = [];
                (data.trees || []).forEach((t, idx) => {
                    const arr = (t.components && (t.components[String(dim)] || t.components[dim])) || [];
                    const split = (!identified && t.split_event && (t.split_event.col === dim || t.split_event.col === String(dim))) ? t.split_event.split_value : null;
                    const label = (!identified && t.split_event && (t.split_event.col === dim || t.split_event.col === String(dim))) ? `T${t.tree_id}` : '';
                    comps.push({ intervals: arr, colorIndex: idx, tree_id: t.tree_id, key: `tree_${t.tree_id}`, split_value: split, split_label: label });
                });

                const mergedTitle = identified ? `Feature ${dim} (Merged and Identified)` : `Feature ${dim} (Merged)`;
                Charts.stepFunctions(
                    el,
                    comps.map((c, i) => ({ intervals: c.intervals, color: Charts.__theme?.palette?.[i % (Charts.__theme?.palette?.length || 7)], key: c.key, split_value: c.split_value, split_label: c.split_label })),
                    bounds,
                    { title: mergedTitle, hoverable: true, annotate: true, log: useLog, modalZoom: true }
                );

                if (combinedSnapshot) {
                    overlayCombinedComponent(el, combinedSnapshot, dim, bounds, useLog, comps);
                }

                // Add hover interactions
                const svg = d3.select(el).select('svg');
                const tooltip = Charts.getTooltip();
                createHoverInteractions(svg, tooltip, bounds, comps, treeColors, combinedSnapshot, dim, useLog);
            }
            return;
        }
        treeSelect.disabled = false;

        // Handle multiple selected trees or single tree
        if (selectedTrees.length === 1) {
            // Single tree mode
            const treeId = selectedTrees[0];
            const data = identified2
                ? await (async () => {
                    // Build a synthetic evolution object from identified components endpoint
                    const idc = await API.getIdentified(runId, epoch, treeId);
                    const n = idc.n_dims || 0;
                    const estimators = [];
                    for (let d = 0; d < n; d++) {
                        const comp = idc.components && (idc.components[String(d)] || idc.components[d]);
                        if (comp && comp.intervals && comp.values) {
                            const est = comp.intervals.map((ab, i) => [ab[0], ab[1], comp.values[i]]);
                            estimators.push(est);
                        } else {
                            estimators.push([[-Infinity, Infinity, 1.0]]);
                        }
                    }
                    // Build bounds from intervals
                    const bounds = {};
                    for (let d = 0; d < n; d++) {
                        const comp = idc.components && (idc.components[String(d)] || idc.components[d]);
                        if (comp && comp.intervals && comp.intervals.length) {
                            const finite = comp.intervals.flat().filter(v => isFinite(v));
                            if (finite.length) {
                                const mn = Math.min(...finite);
                                const mx = Math.max(...finite);
                                const m = (mx - mn) > 0 ? 0.1 * (mx - mn) : 1;
                                bounds[d] = { min: mn - m, max: mx + m };
                                continue;
                            }
                        }
                        bounds[d] = { min: -1, max: 1 };
                    }
                    return { n_dims: n, estimators, bounds, split_event: null };
                })()
                : await API.getTreeEvolution(runId, epoch, treeId, iteration);
            nDims = data.n_dims || (data.estimators ? data.estimators.length : 0) || 0;
            if (!nDims || !data.estimators) return;

            // Setup grid layout
            const { needsRebuild, cols } = setupGridLayout(container, nDims, lastLayout, runId, epoch, false);
            rebuildGrid = needsRebuild;
            if (needsRebuild) {
                lastLayout = { merged: false, runId, epoch, nDims, cols };
            }

            // Update event metrics - include current_error from data if available
            if (data.split_event && data.current_error !== undefined) {
                data.split_event.current_error = data.current_error;
            }
            updateEventMetrics(data.split_event, true);

            // Render single tree components
            for (let dim = 0; dim < nDims; dim++) {
                const { el, bounds } = renderChartForDimension(container, dim, nDims, data, data.bounds, identified2, useLog, combinedSnapshot, {}, rebuildGrid);

                const split = (data.split_event && (data.split_event.col === dim || data.split_event.col === String(dim))) ? data.split_event.split_value : null;
                const label = (data.split_event && (data.split_event.col === dim || data.split_event.col === String(dim))) ? `${String(data.split_event.action || '').toLowerCase()} @ ${Number(split).toPrecision(3)}` : '';
                const singleTitle = identified2 ? `Feature ${dim} (Identified)` : `Feature ${dim}`;
                const singleComp = [{ intervals: data.estimators[dim], key: 'tree_current', split_value: split, split_label: label }];

                Charts.stepFunctions(el, singleComp, bounds, { title: singleTitle, annotate: true, log: useLog, modalZoom: true });

                if (combinedSnapshot) {
                    overlayCombinedComponent(el, combinedSnapshot, dim, bounds, useLog, singleComp);
                }

                // Add hover interactions (single tree version)
                const svg = d3.select(el).select('svg');
                const tooltip = Charts.getTooltip();
                createHoverInteractions(svg, tooltip, bounds, singleComp, {}, combinedSnapshot, dim, useLog);
            }
        } else {
            // Multiple trees mode - use the unified endpoint
            const identified = document.getElementById('identifiedToggle')?.checked === true;
            const data = await API.getUnifiedTreeComponents(runId, epoch, iteration, identified, selectedTrees.join(','));

            // No need to filter here since the API now returns only the selected trees

            nDims = data.n_dims || 0;
            if (!nDims) return;

            // Setup grid layout
            const { needsRebuild, cols } = setupGridLayout(container, nDims, lastLayout, runId, epoch, false);
            rebuildGrid = needsRebuild;
            if (needsRebuild) {
                lastLayout = { merged: false, runId, epoch, nDims, cols };
            }

            // Update event metrics
            const latestEvent = findLatestSplitEvent(data.trees);
            updateEventMetrics(latestEvent, false);

            // Setup tree colors
            const palette = (Charts.__theme && Charts.__theme.palette) || [];
            const treeColors = {};
            (data.trees || []).forEach((t, i) => {
                treeColors[`tree_${t.tree_id}`] = palette[i % (palette.length || 7)] || '#555';
            });

            // Render multiple trees
            for (let dim = 0; dim < nDims; dim++) {
                const { el, bounds } = renderChartForDimension(container, dim, nDims, data, data.bounds, identified, useLog, combinedSnapshot, treeColors, rebuildGrid);

                const comps = [];
                (data.trees || []).forEach((t, idx) => {
                    // Now both identified and regular components use the same format
                    const arr = (t.components && t.components[dim]) || [];

                    // Only add components that have valid intervals
                    if (arr && arr.length > 0) {
                        const split = (!identified && t.split_event && (t.split_event.col === dim || t.split_event.col === String(dim))) ? t.split_event.split_value : null;
                        const label = (!identified && t.split_event && (t.split_event.col === dim || t.split_event.col === String(dim))) ? `T${t.tree_id}` : '';
                        comps.push({ intervals: arr, colorIndex: idx, tree_id: t.tree_id, key: `tree_${t.tree_id}`, split_value: split, split_label: label });
                    }
                });

                const multiTitle = identified ? `Feature ${dim} (Selected Trees - Identified)` : `Feature ${dim} (Selected Trees)`;

                // Prepare data for stepFunctions
                const stepData = comps.map((c, i) => ({ intervals: c.intervals, color: Charts.__theme?.palette?.[i % (Charts.__theme?.palette?.length || 7)], key: c.key, split_value: c.split_value, split_label: c.split_label }));

                // Only render if we have valid components
                if (stepData.length > 0) {
                    Charts.stepFunctions(
                        el,
                        stepData,
                        bounds,
                        { title: multiTitle, hoverable: true, annotate: true, log: useLog, modalZoom: true }
                    );
                } else {
                    // Show a message when no data is available
                    el.innerHTML = `<div style="text-align: center; padding: 20px; color: #666;">
                        <p>No component data available for Feature ${dim}</p>
                        <p>Selected trees: ${selectedTrees.join(', ')}</p>
                    </div>`;
                }

                if (combinedSnapshot) {
                    overlayCombinedComponent(el, combinedSnapshot, dim, bounds, useLog, comps);
                }

                // Add hover interactions
                const svg = d3.select(el).select('svg');
                const tooltip = Charts.getTooltip();
                createHoverInteractions(svg, tooltip, bounds, comps, treeColors, combinedSnapshot, dim, useLog);
            }
        }
    }

    runSelect.addEventListener('change', loadEpochsTrees);
    epochSelect.addEventListener('change', async () => { await refreshTreeList(); await updateIterationBounds(); });
    treeSelect.addEventListener('change', async () => { await updateIterationBounds(); render(); });

    // Handle multi-select and ensure render updates
    treeSelect.addEventListener('change', async (e) => {
        console.log('Tree selection changed');
        const selectedTrees = Array.from(treeSelect.selectedOptions).map(opt => +opt.value);
        console.log('Selected trees after change:', selectedTrees);
        // Hide selection info when user manually changes selection
        const selectionInfo = document.getElementById('selectionInfo');
        if (selectionInfo) {
            const selectedCount = selectedTrees.length;
            selectionInfo.textContent = `Showing ${selectedCount} selected trees from error plot selection`;
            selectionInfo.style.display = 'block';
            // Hide event metrics when showing selection info
            document.getElementById('eventMetrics').style.display = 'none';
        }
        await updateIterationBounds();
        render();
    });

    // Also listen for mouse events to handle multi-select
    treeSelect.addEventListener('mouseup', (e) => {
        if (e.target.tagName === 'OPTION') {
            console.log('Option mouseup:', e.target.value, 'selected:', e.target.selected);
            // Small delay to ensure DOM is updated
            setTimeout(() => {
                const selectedTrees = Array.from(treeSelect.selectedOptions).map(opt => +opt.value);
                console.log('Selected trees after mouseup:', selectedTrees);
                // Hide selection info when user manually changes selection
                const selectionInfo = document.getElementById('selectionInfo');
                if (selectionInfo) {
                    selectionInfo.style.display = 'none';
                    // Show event metrics when hiding selection info
                    document.getElementById('eventMetrics').style.display = 'flex';
                }
                updateIterationBounds().then(() => render());
            }, 50);
        }
    });


    iterationRange.addEventListener('input', () => {
        iterationValue.textContent = iterationRange.value;
        // Live update to ensure multi-select renders on drag
        render();
    });
    iterationRange.addEventListener('change', render);
    mergedToggle.addEventListener('change', async () => {
        const on = mergedToggle.checked === true;
        if (on) {
            // Clear any selected trees so merged mode shows all trees
            try {
                Array.from(treeSelect.options).forEach(opt => { opt.selected = false; });
            } catch { }
            // Hide selection info when switching to merged
            const selectionInfo = document.getElementById('selectionInfo');
            if (selectionInfo) {
                selectionInfo.style.display = 'none';
            }
            // Show event metrics container; it will be updated in render
            const em = document.getElementById('eventMetrics');
            if (em) {
                em.style.display = 'flex';
            }
        } else {
            // Leaving merged mode: immediately enable tree selector and ensure at least one tree is selected
            treeSelect.disabled = false;
            // If nothing selected, select the first available tree to avoid early-return in render
            const selectedCount = Array.from(treeSelect.selectedOptions).length;
            if (selectedCount === 0 && treeSelect.options.length > 0) {
                treeSelect.selectedIndex = 0;
            }
            // Hide selection info (user can multi-select after) and show metrics
            const selectionInfo = document.getElementById('selectionInfo');
            if (selectionInfo) {
                selectionInfo.style.display = 'none';
            }
            const em = document.getElementById('eventMetrics');
            if (em) {
                em.style.display = 'flex';
            }
        }
        await updateIterationBounds();
        render();
    });
    if (combinedOverlayToggle) {
        combinedOverlayToggle.addEventListener('change', () => { render(); });
    }
    const logToggleEl = document.getElementById('logToggle');
    if (logToggleEl) {
        logToggleEl.addEventListener('change', () => { render(); });
    }
    const idToggleEl = document.getElementById('identifiedToggle');
    if (idToggleEl) {
        idToggleEl.addEventListener('change', async () => {
            // Disable iteration slider when identified
            const on = idToggleEl.checked === true;
            iterationRange.disabled = on;
            if (on) {
                // Jump to last iteration so unchecking shows last state
                await updateIterationBounds();
                const maxIt = Number(iterationRange.max || 0);
                iterationRange.value = String(maxIt);
                iterationValue.textContent = String(maxIt);
            }
            render();
        });
    }

    // Keyboard navigation with ArrowLeft/ArrowRight
    document.addEventListener('keydown', (ev) => {
        if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
        // Avoid interfering with text inputs
        const tag = (ev.target && ev.target.tagName) ? ev.target.tagName.toLowerCase() : '';
        if (tag === 'input' || tag === 'select' || tag === 'textarea') {
            // allow slider arrows to work natively too
            if (ev.target !== iterationSlider) return;
        }
        const min = Number(iterationSlider.min || 0);
        const max = Number(iterationSlider.max || 0);
        let v = Number(iterationSlider.value || 0);
        if (ev.key === 'ArrowRight') v = Math.min(max, v + 1);
        if (ev.key === 'ArrowLeft') v = Math.max(min, v - 1);
        if (v !== Number(iterationSlider.value)) {
            iterationSlider.value = String(v);
            iterationValue.textContent = String(v);
            render();
            ev.preventDefault();
        }
    });

    loadEpochsTrees();

    // Clear event metrics on page load
    clearEventMetrics();

    // Re-render on resize/zoom to adapt axes text and tiles per row
    let resizeTimer = null;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => { render(); }, 120);
    });

    // parsing moved to UI.toFloat

    // Overlay the combined-product component as a dashed line with thicker stroke.
    function overlayCombinedComponent(containerEl, snapshot, dim, bounds, useLog, baseComps) {
        try {
            const vals = (snapshot.snapshot.grid_values || [])[dim] || [];
            const ilist = (snapshot.snapshot.intervals || [])[dim] || [];
            if (!ilist.length || !vals.length) return;
            // Use the same transform domain as the base chart: if base comps are all-zero and log is on, base rendered values are 1
            const allZeroBase = Array.isArray(baseComps) && baseComps.every(c => (c.intervals || []).every(iv => Math.abs(iv[2] || 0) < 1e-12));
            const intervals = ilist.map((ib, i) => [UI.toFloat(ib[0]), UI.toFloat(ib[1]), UI.toFloat(vals[i])]).filter(v => v.length === 3);
            const { width, height, margin, innerW, innerH } = Charts.sizeOf(containerEl);
            const svg = d3.select(containerEl).select('svg');
            // Reuse same scales as main chart by recomputing with same bounds and log flag
            const [lo, hi] = [bounds.min, bounds.max];
            const x = d3.scaleLinear().domain([lo, hi]).range([0, innerW]);
            function sy(v) { return Math.sign(v) * Math.log10(1 + Math.abs(v)); }
            const eps = 1e-12;
            const arr = intervals.map(iv => {
                const v = iv[2] == null ? 0 : +iv[2];
                // Match base log rendering: if base was all zero, baseline was 1 under log
                if (useLog && allZeroBase) return sy(1);
                return useLog ? sy(v) : v;
            });
            // Compute Y scale from base components to match the existing chart
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
            if (!isFinite(ymin) || !isFinite(ymax)) { ymin = -1; ymax = 1; }
            if (ymin === ymax) { ymin -= 1; ymax += 1; }
            const y = d3.scaleLinear().domain([ymin, ymax]).nice().range([innerH, 0]);
            const points = [];
            intervals.forEach(([a, b, _v], i) => {
                const left = Number.isFinite(a) ? a : lo;
                const right = Number.isFinite(b) ? b : hi;
                const vy = arr[i];
                points.push([left, vy], [right, vy]);
            });
            if (points.length >= 2) {
                const line = d3.line().x(d => x(d[0])).y(d => y(d[1])).curve(d3.curveStepAfter);
                svg.select('g')
                    .append('path')
                    .datum(points)
                    .attr('fill', 'none')
                    .attr('stroke', '#111')
                    .attr('stroke-width', 1.4 * 1.5)
                    .attr('stroke-dasharray', '4,3')
                    .attr('opacity', 0.9)
                    .attr('d', line)
                    .attr('class', 'trace primary-overlay')
                    .attr('data-key', 'combined_overlay');
            }
        } catch { }
    }

    // Function to clear event metrics display
    function clearEventMetrics() {
        const eventMetrics = document.getElementById('eventMetrics');
        if (eventMetrics) {
            eventMetrics.style.display = 'none';
        }
    }

    // Initial render
    render();
})();
