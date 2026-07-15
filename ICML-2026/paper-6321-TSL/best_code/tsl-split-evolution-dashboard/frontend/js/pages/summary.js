(async function () {
    try {
        const runs = await API.getRuns();
        // Populate dropdown
        const runSelect = document.getElementById('runSelectSummary');
        runSelect.innerHTML = '';
        runs.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r.run_id;
            opt.textContent = `Run ${r.run_id} (${r.n_events} events)`;
            runSelect.appendChild(opt);
        });
        // Restore previously selected run from URL or localStorage
        const params = new URLSearchParams(window.location.search);
        const urlRun = params.get('run');
        const storedRun = localStorage.getItem('selectedRunId');
        const preferred = [urlRun, storedRun, runs.length ? String(runs[0].run_id) : null].find(Boolean);
        if (preferred && Array.from(runSelect.options).some(o => o.value === String(preferred))) {
            runSelect.value = String(preferred);
        } else if (runs.length > 0) {
            runSelect.value = String(runs[0].run_id);
        }

        function updateNavLinks(runId) {
            const links = document.querySelectorAll('.nav a');
            links.forEach(a => {
                try {
                    const u = new URL(a.href, window.location.origin);
                    if (u.pathname === '/' || u.pathname.endsWith('/')) {
                        // keep run on summary too
                        if (runId) u.searchParams.set('run', runId);
                    }
                    if (u.pathname.endsWith('/tree.html') || u.pathname.endsWith('/primary.html')) {
                        if (runId) u.searchParams.set('run', runId);
                    }
                    a.href = u.pathname + (u.search || '');
                } catch { }
            });
        }

        function renderOverview() {
            const ev = document.getElementById('summaryEvents');
            const sp = document.getElementById('summarySplits');
            const it = document.getElementById('summaryIterations');
            const co = document.getElementById('summaryColumns');
            if (!ev || !sp || !it || !co) return; // Overview block was removed
            Charts.barChart(ev, runs.map(r => ({ id: r.run_id, events: r.n_events })), { xKey: 'id', yKey: 'events', title: 'Events per Run' });
            Charts.stackedBars(sp, runs.map(r => ({ id: r.run_id, split: r.n_splits, resplit: r.n_resplits })), 'id', ['split', 'resplit'], ['#1B998B', '#F18F01'], 'Splits vs Resplits');
            Charts.lineChart(it, runs.map(r => ({ id: r.run_id, iters: r.n_iterations })).sort((a, b) => a.id - b.id), { xKey: 'id', yKey: 'iters', title: 'Iterations per Run', color: '#C73E1D' });
            Charts.barChart(co, runs.map(r => ({ id: r.run_id, cols: r.n_columns_split })), { xKey: 'id', yKey: 'cols', title: 'Columns Split' });
        }

        async function renderRunSpecific() {
            const runId = +runSelect.value;
            if (!Number.isFinite(runId)) return;
            localStorage.setItem('selectedRunId', String(runId));
            updateNavLinks(String(runId));
            // Compute summary counts for splits/resplits/merges
            const tl = await API.getTimeline(runId);
            const splitCount = tl.filter(e => String(e.action).toLowerCase() === 'split').length;
            const resplitCount = tl.filter(e => String(e.action).toLowerCase() === 'resplit').length;
            const mergeCount = tl.filter(e => String(e.action).toLowerCase() === 'merge').length;
            const statsEl = document.getElementById('runStats');
            if (statsEl) {
                statsEl.innerHTML = `
                    <div class="metric">Splits: <b>${splitCount}</b></div>
                    <div class="metric">Resplits: <b>${resplitCount}</b></div>
                    <div class="metric">Merges: <b>${mergeCount}</b></div>
                `;
            }
            // Learning: render training and test errors using grid_errors endpoint
            async function renderErrorsPlots() {
                const containerTrain = document.getElementById('trainErrTrain');
                const containerTest = document.getElementById('trainErrTest');
                if (!containerTrain && !containerTest) return;
                const ge = await API.getGridErrors(runId);
                const useLog = document.getElementById('logScaleToggle')?.checked === true;

                // Helper to draw one panel: tree scatter + family line overlay
                function drawPanel(container, treePts, familyVals, title, color) {
                    if (!container) return;
                    const errorType = title.includes('Training') ? 'Individual training error' : 'Individual test error';

                    // Filter out outliers: keep only errors in 95% range (remove top 5%)
                    let filteredTreePts = treePts;
                    if (treePts.length > 0) {
                        // Extract all error values and sort them
                        const errors = treePts.map(d => +d.err).filter(e => isFinite(e));
                        if (errors.length > 0) {
                            errors.sort((a, b) => a - b);
                            // Calculate 95th percentile (index at 95% of sorted array)
                            const percentile95Index = Math.floor(errors.length * 0.95);
                            const percentile95Value = errors[percentile95Index];
                            // Filter out points with errors above 95th percentile
                            filteredTreePts = treePts.filter(d => +d.err <= percentile95Value);
                        }
                    }

                    const grouped = [{
                        name: 'trees',
                        color: '#8C8C8C',
                        data: filteredTreePts.map((d, index) => {
                            // Add jitter to epoch to spread out overlapping points
                            const jitterAmount = 0.15; // Adjust this value to control spread
                            const jitteredEpoch = (d.epoch ?? 0) + (Math.random() - 0.5) * jitterAmount;

                            return {
                                epoch: jitteredEpoch,  // Used for positioning
                                err: +d.err,
                                originalEpoch: d.epoch ?? 0,  // Store original epoch separately
                                tooltipData: {
                                    'Tree ID': d.tree_id || 'N/A',
                                    'Epoch': d.epoch ?? 0  // Show original epoch in tooltip
                                }
                            };
                        }),
                        tooltipTitle: errorType
                    }];
                    Charts.scatterWithSelection(container, grouped, {
                        xKey: 'epoch',
                        yKey: 'err',
                        title,
                        log: useLog,
                        onSelectionChange: (selectedPoints) => {
                            // Navigate to tree evolution tab with selected trees
                            if (selectedPoints.length > 0) {
                                const treeIds = selectedPoints.map(p => p.tooltipData?.['Tree ID'] || p.tree_id).filter(id => id !== 'N/A');
                                if (treeIds.length > 0) {
                                    const runId = +runSelect.value;
                                    const epoch = selectedPoints[0].originalEpoch ?? selectedPoints[0].epoch;
                                    const treeIdsParam = treeIds.join(',');

                                    // Show navigation message
                                    const selectionInfo = d3.select(container).select('.selection-info');
                                    if (selectionInfo.size() > 0) {
                                        // Remove any existing navigation message
                                        selectionInfo.select('.nav-message').remove();

                                        const navMessage = selectionInfo.append('div')
                                            .attr('class', 'nav-message')
                                            .style('margin-top', '8px')
                                            .style('padding', '8px')
                                            .style('background', '#fef3c7')
                                            .style('border', '1px solid #f59e0b')
                                            .style('border-radius', '4px')
                                            .style('color', '#92400e')
                                            .style('font-size', '12px')
                                            .style('cursor', 'pointer')
                                            .style('text-align', 'center')
                                            .style('font-weight', '500');
                                        navMessage.text(`Click to open product evolution showing ${treeIds.length} selected products`);
                                        navMessage.on('click', () => {
                                            // Determine dominant epoch in selection (most frequent)
                                            const epochCounts = new Map();
                                            selectedPoints.forEach(p => {
                                                const e = p.originalEpoch ?? +p.epoch;
                                                epochCounts.set(e, (epochCounts.get(e) || 0) + 1);
                                            });
                                            let chosenEpoch = epoch;
                                            if (epochCounts.size > 0) {
                                                chosenEpoch = Array.from(epochCounts.entries()).sort((a, b) => b[1] - a[1])[0][0];
                                            }
                                            // Filter tree IDs to the chosen epoch only
                                            const treesAtEpoch = selectedPoints
                                                .filter(p => (p.originalEpoch ?? +p.epoch) === +chosenEpoch)
                                                .map(p => p.tooltipData?.['Tree ID'] || p.tree_id)
                                                .filter(id => id !== 'N/A');
                                            const treesParam = (treesAtEpoch.length ? treesAtEpoch : treeIds).join(',');
                                            const url = `/pages/tree.html?run=${runId}&epoch=${chosenEpoch}&trees=${treesParam}`;
                                            window.location.href = url;
                                        });
                                    }
                                }
                            }
                        }
                    });
                    if (Array.isArray(ge.epochs) && Array.isArray(familyVals) && ge.epochs.length === familyVals.length) {
                        const svg = d3.select(container).select('svg');
                        const g = svg.select('g');
                        const { innerW, innerH } = (function () { const m = Charts.sizeOf(container); return { innerW: m.innerW, innerH: m.innerH }; })();
                        const epochs = ge.epochs.slice();
                        const famSeries = epochs.map((e, i) => ({ epoch: e, err: familyVals[i] })).filter(d => d.err != null);
                        if (famSeries.length) {
                            const all = grouped[0].data.concat(famSeries);
                            const x = d3.scaleLinear().domain(d3.extent(all, d => +d.epoch)).nice().range([0, innerW]);

                            // Apply log scale to Y if enabled
                            let y;
                            if (useLog) {
                                const yDomainSrc = all.map(d => +d.err);
                                const yTrans = yDomainSrc.map(v => Math.sign(v) * Math.log10(1 + Math.abs(v)));
                                const yExtent = d3.extent(yTrans);
                                y = d3.scaleLinear().domain(yExtent).nice().range([innerH, 0]);
                            } else {
                                y = d3.scaleLinear().domain(d3.extent(all, d => +d.err)).nice().range([innerH, 0]);
                            }

                            const line = d3.line().x(d => x(+d.epoch)).y(d => y(useLog ? Math.sign(d.err) * Math.log10(1 + Math.abs(d.err)) : +d.err)).curve(d3.curveMonotoneX);
                            g.append('path')
                                .datum(famSeries.sort((a, b) => a.epoch - b.epoch))
                                .attr('fill', 'none')
                                .attr('stroke', color)
                                .attr('stroke-width', 2)
                                .attr('d', line);

                            // Add family error points with bigger size and matching color
                            g.selectAll('circle.family')
                                .data(famSeries)
                                .join('circle')
                                .attr('class', 'family')
                                .attr('cx', d => x(+d.epoch))
                                .attr('cy', d => y(useLog ? Math.sign(d.err) * Math.log10(1 + Math.abs(d.err)) : +d.err))
                                .attr('r', 3)
                                .attr('fill', color)
                                .attr('stroke', color === '#C44E52' ? '#7a1f24' : '#1b4e6b')
                                .attr('stroke-width', 1.5)
                                .on('mousemove', function (event, d) {
                                    const tooltip = Charts.getTooltip();
                                    d3.select(this).attr('r', 6);
                                    tooltip.style.display = 'block';
                                    tooltip.style.left = (event.pageX + 12) + 'px';
                                    tooltip.style.top = (event.pageY + 12) + 'px';
                                    const errorType = title.includes('Training') ? 'Combined-product training error' : 'Combined-product test error';
                                    const epochVal = +d.epoch;
                                    const errVal = +d.err;
                                    tooltip.innerHTML = `<b>${errorType}</b><div>Stage: ${Number.isFinite(epochVal) ? epochVal.toLocaleString() : String(d.epoch ?? '')}</div><div>Error: ${Number.isFinite(errVal) ? errVal.toLocaleString() : String(d.err ?? '')}</div>`;
                                })
                                .on('mouseleave', function () {
                                    d3.select(this).attr('r', 5);
                                    const tooltip = Charts.getTooltip();
                                    if (tooltip) tooltip.style.display = 'none';
                                });
                        }
                    }
                }
                drawPanel('#trainErrTrain', ge.trees.train || [], ge.family.train || [], 'Training error', '#C44E52');
                drawPanel('#trainErrTest', ge.trees.test || [], ge.family.test || [], 'Test error', '#2E86AB');
            }
            await renderErrorsPlots();

            // Render energy and scalings plots
            async function renderEnergyAndScalings() {
                // Fetch energy and scalings data
                let energyData = null;
                let scalingsData = null;

                try {
                    energyData = await API.getEpochEnergy(runId);
                } catch (e) {
                    console.warn('Failed to fetch energy data:', e);
                }

                try {
                    scalingsData = await API.getEpochScalings(runId);
                } catch (e) {
                    console.warn('Failed to fetch scalings data:', e);
                }

                // Calculate and plot energy * scaling^2 using latest scaling for each epoch
                if (energyData && scalingsData && energyData.epochs && scalingsData.latest) {
                    const latestScalings = scalingsData.latest; // {epoch: scaling} mapping

                    // Prepare energy data
                    let energySeries = energyData.epochs.map((epoch, idx) => ({
                        epoch,
                        energy: energyData.energy[idx]
                    })).filter(d => d.energy != null);

                    let energyScalingSquared = energyData.epochs.map((epoch, idx) => {
                        const energy = energyData.energy[idx];
                        const scaling = latestScalings[epoch];
                        if (energy != null && scaling != null) {
                            return {
                                epoch,
                                value: energy * scaling * scaling
                            };
                        }
                        return null;
                    }).filter(d => d !== null);

                    // Prepare scalings data
                    let groupedSeries = [];
                    if (scalingsData.scalings && scalingsData.scalings.length > 0) {
                        const byOptEpoch = {};
                        scalingsData.scalings.forEach(s => {
                            const optEpoch = s.optimization_epoch;
                            if (!byOptEpoch[optEpoch]) {
                                byOptEpoch[optEpoch] = [];
                            }
                            byOptEpoch[optEpoch].push({
                                epoch: s.epoch,
                                scaling: s.scaling
                            });
                        });

                        groupedSeries = Object.keys(byOptEpoch)
                            .sort((a, b) => +a - +b)
                            .map((optEpoch, idx) => ({
                                name: `Optimized at stage ${optEpoch}`,
                                color: Charts.__theme.palette[idx % Charts.__theme.palette.length],
                                data: byOptEpoch[optEpoch].sort((a, b) => a.epoch - b.epoch)
                            }));
                    }

                    // Store original data for reset
                    const originalEnergySeries = [...energySeries];
                    const originalEnergyScalingSquared = [...energyScalingSquared];
                    const originalGroupedSeries = groupedSeries.map(s => ({
                        ...s,
                        data: [...s.data]
                    }));

                    function renderAllCharts(useSortedIndex = false) {
                        // Render energy chart
                        const energyEl = document.getElementById('energyChart');
                        if (energyEl && energySeries.length > 0) {
                            const dataToRender = useSortedIndex
                                ? energySeries.map((d, idx) => ({ ...d, displayIndex: idx }))
                                : energySeries;
                            Charts.lineChart(energyEl, dataToRender, {
                                xKey: useSortedIndex ? 'displayIndex' : 'epoch',
                                yKey: 'energy',
                                title: 'Energy per stage',
                                color: '#55A868'
                            });
                        }

                        // Render energy × scaling² chart
                        const energyScalingEl = document.getElementById('energyScalingSquaredChart');
                        if (energyScalingEl && energyScalingSquared.length > 0) {
                            const dataToRender = useSortedIndex
                                ? energyScalingSquared.map((d, idx) => ({ ...d, displayIndex: idx }))
                                : energyScalingSquared;
                            Charts.lineChart(energyScalingEl, dataToRender, {
                                xKey: useSortedIndex ? 'displayIndex' : 'epoch',
                                yKey: 'value',
                                title: 'Energy × scaling² (latest scaling)',
                                color: '#DD8452',
                                log: true
                            });
                        }

                        // Render scalings chart
                        const scalingsEl = document.getElementById('scalingsChart');
                        if (scalingsEl && groupedSeries.length > 0) {
                            const seriesToRender = useSortedIndex
                                ? groupedSeries.map(s => ({
                                    ...s,
                                    data: s.data.map((d, idx) => ({ ...d, displayIndex: idx }))
                                }))
                                : groupedSeries;
                            Charts.multiLineChart(scalingsEl, seriesToRender, {
                                xKey: useSortedIndex ? 'displayIndex' : 'epoch',
                                yKey: 'scaling',
                                title: 'Scaling per stage (grouped by optimization stage)'
                            });
                        }
                    }

                    // Initial render
                    renderAllCharts(false);

                    // Sort toggle handler
                    const sortToggle = document.getElementById('sortEnergyScalingToggle');
                    if (sortToggle) {
                        sortToggle.addEventListener('change', () => {
                            if (sortToggle.checked) {
                                // Sort all data by Energy × Scaling² value in descending order
                                const sortedEpochs = [...energyScalingSquared]
                                    .sort((a, b) => b.value - a.value)
                                    .map(d => d.epoch);

                                // Create epoch to index mapping for sorting
                                const epochToIndex = new Map();
                                sortedEpochs.forEach((epoch, idx) => {
                                    epochToIndex.set(epoch, idx);
                                });

                                // Sort energy series
                                energySeries = [...energySeries].sort((a, b) => {
                                    const idxA = epochToIndex.get(a.epoch) ?? Infinity;
                                    const idxB = epochToIndex.get(b.epoch) ?? Infinity;
                                    return idxA - idxB;
                                });

                                // Sort energyScalingSquared (already sorted by value)
                                energyScalingSquared = [...energyScalingSquared].sort((a, b) => b.value - a.value);

                                // Sort scalings data for each series
                                groupedSeries = groupedSeries.map(s => ({
                                    ...s,
                                    data: [...s.data].sort((a, b) => {
                                        const idxA = epochToIndex.get(a.epoch) ?? Infinity;
                                        const idxB = epochToIndex.get(b.epoch) ?? Infinity;
                                        return idxA - idxB;
                                    })
                                }));
                            } else {
                                // Restore original order
                                energySeries = [...originalEnergySeries];
                                energyScalingSquared = [...originalEnergyScalingSquared];
                                groupedSeries = originalGroupedSeries.map(s => ({
                                    ...s,
                                    data: [...s.data]
                                }));
                            }
                            renderAllCharts(sortToggle.checked);
                        });
                    }
                }

            }
            await renderEnergyAndScalings();

            const timeline = await API.getTimeline(runId);
            const countsEl = document.getElementById('timelineCounts');
            const scatterEl = document.getElementById('timelineScatter');
            if (countsEl) {
                const byIter = d3.rollups(timeline, v => v.length, d => d.iter_no).map(([iter, cnt]) => ({ iter, cnt })).sort((a, b) => a.iter - b.iter);
                Charts.barChart(countsEl, byIter, { xKey: 'iter', yKey: 'cnt', title: 'Events per Iteration', color: '#1B998B' });
            }
            if (scatterEl) {
                const filtered = timeline.filter(d => d.split_value != null && isFinite(+d.split_value));
                const groups = d3.groups(filtered, d => d.col).map(([col, arr]) => ({ name: `col-${col}`, data: arr.map(r => ({ iter_no: r.iter_no, split_value: +r.split_value })) }));
                Charts.scatter(scatterEl, groups, { xKey: 'iter_no', yKey: 'split_value', title: 'Split Values over Iterations (by Column)' });
            }

            const colStats = await API.getColumns(runId);
            const events = colStats.map(d => ({ col: d.col, splits: d.n_splits, resplits: d.n_resplits }));
            const avgGain = colStats.map(d => ({ col: d.col, avg: d.avg_gain }));
            const samples = colStats.map(d => ({ col: d.col, val: d.avg_samples_affected }));
            const colEventsEl = document.getElementById('colEvents');
            const colAvgGainEl = document.getElementById('colAvgGain');
            const colSamplesEl = document.getElementById('colSamples');
            if (colEventsEl) Charts.stackedBars(colEventsEl, events, 'col', ['splits', 'resplits'], ['#1B998B', '#F18F01'], 'Events per Column');
            if (colAvgGainEl) Charts.barChart(colAvgGainEl, avgGain, { xKey: 'col', yKey: 'avg', title: 'Average Gain per Column', color: '#A23B72' });
            if (colSamplesEl) Charts.barChart(colSamplesEl, samples, { xKey: 'col', yKey: 'val', title: 'Average Samples Affected', color: '#2E86AB' });

            const conv = await API.getConvergence(runId);
            const cepochs = (conv && conv.epochs) || [];
            const cseries = cepochs.map((e, i) => ({
                epoch: e,
                total: (conv.total || [])[i] ?? 0,
                avg: (conv.avg || [])[i] ?? 0,
                new_splits: (conv.new_splits || [])[i] ?? 0,
                resplits: (conv.resplits || [])[i] ?? 0,
                merges: (conv.merges || [])[i] ?? 0,
                complexity: (conv.complexity || [])[i] ?? 0
            }));
            const convTotalEl = document.getElementById('convTotal');
            const convAvgEl = document.getElementById('convAvg');
            const convKindsEl = document.getElementById('convSplitKinds');
            const convCxEl = document.getElementById('convComplexity');
            if (convTotalEl) Charts.lineChart(convTotalEl, cseries, { xKey: 'epoch', yKey: 'total', title: 'Total error reduction', color: '#2E86AB' });
            if (convAvgEl) Charts.lineChart(convAvgEl, cseries, { xKey: 'epoch', yKey: 'avg', title: 'Average gain per split', color: '#A23B72' });
            if (convKindsEl) Charts.stackedBars(convKindsEl, cseries, 'epoch', ['new_splits', 'resplits', 'merges'], ['#2E86AB', '#F18F01', '#937860'], 'Splits / resplits / merges per stage');
            if (convCxEl) Charts.lineChart(convCxEl, cseries, { xKey: 'epoch', yKey: 'complexity', title: 'Model complexity (cells)', color: '#7209B7' });

            // Training/test errors handled in learning section above
        }

        // Render overview only if its container exists
        renderOverview();
        await renderRunSpecific();
        runSelect.addEventListener('change', renderRunSpecific);

        // Add log scale toggle listener
        const logToggle = document.getElementById('logScaleToggle');
        if (logToggle) {
            logToggle.addEventListener('change', () => {
                renderRunSpecific();
            });
        }
    } catch (err) {
        console.error(err);
    }
})();
