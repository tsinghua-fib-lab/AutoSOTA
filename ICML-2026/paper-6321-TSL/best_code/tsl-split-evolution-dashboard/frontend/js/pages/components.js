(async function () {
    const componentRun = document.getElementById('componentRun');
    const componentGrid = document.getElementById('componentGrid');
    const componentGridContainer = document.getElementById('componentGridContainer');
    const logScaleToggle = document.getElementById('logScaleToggle');
    const lambdaGradientToggle = document.getElementById('lambdaGradientToggle');
    const backboneTiltToggle = document.getElementById('backboneTiltToggle');

    if (!componentRun || !componentGrid) {
        console.error('Required elements not found');
        return;
    }

    let allEpochs = [];
    let currentRunId = null;
    let epochData = {}; // Cache: epoch -> data
    let maxComponents = 0;

    async function loadRuns() {
        try {
            const runs = await API.getRuns();
            componentRun.innerHTML = '';
            runs.forEach(r => {
                const opt = document.createElement('option');
                opt.value = r.run_id;
                opt.textContent = `Run ${r.run_id}`;
                componentRun.appendChild(opt);
            });
            if (runs.length > 0) {
                componentRun.value = runs[0].run_id;
                await loadEpochs();
            }
        } catch (err) {
            console.error('Failed to load runs:', err);
        }
    }

    async function loadEpochs() {
        try {
            allEpochs = await API.getCombinedEpochs();
            allEpochs.sort((a, b) => a - b);

            if (allEpochs.length === 0) {
                componentGrid.innerHTML = '<p>No epochs available</p>';
                return;
            }

            await render();
        } catch (err) {
            console.error('Failed to load epochs:', err);
        }
    }

    async function fetchEpochData(epoch) {
        if (epochData[epoch]) {
            return epochData[epoch];
        }
        try {
            const res = await API.getCombinedForEpoch(epoch);
            const runData = res.find(r => r.run_id === parseInt(currentRunId));
            if (runData && runData.snapshot && runData.snapshot.grid_values) {
                epochData[epoch] = runData;
                return runData;
            }
        } catch (err) {
            console.error(`Failed to fetch epoch ${epoch}:`, err);
        }
        return null;
    }

    // Compute f+ and f- from backbone and tilt values per component
    function computeFComponents(snapshot, compIdx) {
        const backbone = (snapshot.backbone_values || [])[compIdx] || [];
        const tilt = (snapshot.tilt_values || [])[compIdx] || [];

        // Compute from backbone and tilt (without lambda scaling)
        if (backbone.length === 0 || tilt.length === 0 || backbone.length !== tilt.length) {
            return { f_plus: null, f_minus: null };
        }

        const f_plus = backbone.map((b, i) => {
            const d = tilt[i];
            const exp_d = Math.min(d, 50.0);
            return Math.max(b, 1e-10) * Math.exp(exp_d);
        });

        const f_minus = backbone.map((b, i) => {
            const d = tilt[i];
            const exp_neg_d = Math.min(-d, 50.0);
            return Math.max(b, 1e-10) * Math.exp(exp_neg_d);
        });

        return { f_plus, f_minus };
    }

    async function render() {
        const runId = componentRun.value;
        if (!runId || allEpochs.length === 0) {
            componentGrid.innerHTML = '';
            return;
        }

        if (currentRunId !== runId) {
            currentRunId = runId;
            epochData = {}; // Clear cache when run changes
        }

        // Display all epochs at once
        const visibleEpochsList = allEpochs;

        // Fetch data for all visible epochs
        const epochDataList = await Promise.all(
            visibleEpochsList.map(epoch => fetchEpochData(epoch))
        );

        // Determine max components across all epochs
        maxComponents = 0;
        epochDataList.forEach(data => {
            if (data && data.snapshot && data.snapshot.grid_values) {
                maxComponents = Math.max(maxComponents, data.snapshot.grid_values.length);
            }
        });

        if (maxComponents === 0) {
            componentGrid.innerHTML = '<p>No component data available</p>';
            return;
        }

        // Clear and set up grid
        componentGrid.innerHTML = '';
        const showBackboneTilt = backboneTiltToggle ? backboneTiltToggle.checked : false;
        // Grid: 1 column for component labels + (2 or 4 columns per epoch) depending on mode
        // When backbone/tilt mode: backbone, tilt, tanh tilt, 2×backbone×sinh(tilt)
        const colsPerEpoch = showBackboneTilt ? 4 : 2;
        componentGrid.style.gridTemplateColumns = `120px repeat(${visibleEpochsList.length * colsPerEpoch}, 400px)`;
        // Two header rows: one for epoch labels (spanning both columns), one for f+/f- labels
        componentGrid.style.gridTemplateRows = `auto auto repeat(${maxComponents}, 300px)`;

        // Create header row 1: empty cell for component label column, then epoch headers spanning both f+ and f-
        const emptyHeader = document.createElement('div');
        emptyHeader.style.cssText = 'padding: 8px; background: #f8f9fa; border: 1px solid #dee2e6; position: sticky; top: 0; left: 0; z-index: 15;';
        componentGrid.appendChild(emptyHeader);

        visibleEpochsList.forEach((epoch, colIdx) => {
            const data = epochDataList[colIdx];

            // Create epoch header cell spanning columns (2 for f+/f-, 4 for backbone/tilt/tanh tilt/2×backbone×sinh(tilt))
            const epochHeader = document.createElement('div');
            epochHeader.style.cssText = `padding: 8px; background: #f8f9fa; border: 1px solid #dee2e6; font-weight: bold; text-align: center; position: sticky; top: 0; z-index: 10; font-size: 0.85rem; line-height: 1.3; grid-column: span ${colsPerEpoch};`;
            
            // Compute lambda asymmetry: (scaling_+ * λ_+ - scaling_- * λ_-) / (scaling_+ * λ_+ + scaling_- * λ_-)
            // This uses the scaled lambdas that reflect the actual contribution to predictions
            let lambdaAsymmetry = 'N/A';
            if (data && data.snapshot) {
                const snapshot = data.snapshot;
                const lambdaPlus = snapshot.lambda_plus;
                const lambdaMinus = snapshot.lambda_minus;
                const scalingPlus = data.scaling_plus != null && isFinite(data.scaling_plus) ? data.scaling_plus : 1.0;
                const scalingMinus = data.scaling_minus != null && isFinite(data.scaling_minus) ? data.scaling_minus : 0.0;
                
                if (lambdaPlus != null && lambdaMinus != null && 
                    isFinite(lambdaPlus) && isFinite(lambdaMinus)) {
                    // Use scaled lambdas: scaling_plus * lambda_plus and scaling_minus * lambda_minus
                    const scaledLambdaPlus = scalingPlus * lambdaPlus;
                    const scaledLambdaMinus = scalingMinus * lambdaMinus;
                    const sum = scaledLambdaPlus + scaledLambdaMinus;
                    if (sum !== 0 && isFinite(sum)) {
                        const asymmetry = (scaledLambdaPlus - scaledLambdaMinus) / sum;
                        lambdaAsymmetry = asymmetry.toFixed(4);
                    }
                }
            }
            
            epochHeader.innerHTML = `
                <div style="font-weight: 600; margin-bottom: 2px;">Stage ${epoch}</div>
                <div style="font-size: 0.75rem; color: #6b7280; margin-top: 2px;">&lambda; asymmetry: ${lambdaAsymmetry}</div>
            `;
            componentGrid.appendChild(epochHeader);
        });

        // Create header row 2: empty cell, then f+ and f- labels with scaling values for each epoch
        // Calculate approximate height of first header row for sticky positioning
        // First row has padding 8px top/bottom + content (~40px) = ~56px
        const firstRowHeight = 56;

        const emptySubHeader = document.createElement('div');
        emptySubHeader.style.cssText = `padding: 8px; background: #f8f9fa; border: 1px solid #dee2e6; position: sticky; top: ${firstRowHeight}px; left: 0; z-index: 15;`;
        componentGrid.appendChild(emptySubHeader);

        visibleEpochsList.forEach((epoch, colIdx) => {
            const data = epochDataList[colIdx];

            if (showBackboneTilt) {
                // Backbone/Tilt mode: show backbone, tilt, and tanh tilt headers
                const headerCellBackbone = document.createElement('div');
                headerCellBackbone.style.cssText = `padding: 8px; background: #f8f9fa; border: 1px solid #dee2e6; font-weight: bold; text-align: center; position: sticky; top: ${firstRowHeight}px; z-index: 10; font-size: 0.85rem; line-height: 1.3;`;
                headerCellBackbone.innerHTML = `
                    <div style="color: #10b981; font-weight: 600; margin-bottom: 2px;">Backbone</div>
                `;
                componentGrid.appendChild(headerCellBackbone);

                const headerCellTilt = document.createElement('div');
                headerCellTilt.style.cssText = `padding: 8px; background: #f8f9fa; border: 1px solid #dee2e6; font-weight: bold; text-align: center; position: sticky; top: ${firstRowHeight}px; z-index: 10; font-size: 0.85rem; line-height: 1.3;`;
                headerCellTilt.innerHTML = `
                    <div style="color: #8b5cf6; font-weight: 600; margin-bottom: 2px;">Tilt</div>
                `;
                componentGrid.appendChild(headerCellTilt);

                const headerCellTanhTilt = document.createElement('div');
                headerCellTanhTilt.style.cssText = `padding: 8px; background: #f8f9fa; border: 1px solid #dee2e6; font-weight: bold; text-align: center; position: sticky; top: ${firstRowHeight}px; z-index: 10; font-size: 0.85rem; line-height: 1.3;`;
                headerCellTanhTilt.innerHTML = `
                    <div style="color: #f59e0b; font-weight: 600; margin-bottom: 2px;">Tanh Tilt</div>
                `;
                componentGrid.appendChild(headerCellTanhTilt);

                const headerCellBackboneSinhTilt = document.createElement('div');
                headerCellBackboneSinhTilt.style.cssText = `padding: 8px; background: #f8f9fa; border: 1px solid #dee2e6; font-weight: bold; text-align: center; position: sticky; top: ${firstRowHeight}px; z-index: 10; font-size: 0.85rem; line-height: 1.3;`;
                headerCellBackboneSinhTilt.innerHTML = `
                    <div style="color: #ec4899; font-weight: 600; margin-bottom: 2px;">2×backbone×sinh(tilt)</div>
                `;
                componentGrid.appendChild(headerCellBackboneSinhTilt);
            } else {
                // f+/f- mode: show f+ and f- headers with scaling and energy
                const scalingPlus = data && data.scaling_plus !== undefined && data.scaling_plus !== null
                    ? data.scaling_plus.toFixed(4)
                    : 'N/A';
                const scalingMinus = data && data.scaling_minus !== undefined && data.scaling_minus !== null
                    ? data.scaling_minus.toFixed(4)
                    : 'N/A';

                // Compute energy_plus and energy_minus from product tensor (without lambda)
                // Energy = mean of squared product tensor: mean(G²) where G = prod_j a_{+,j} (no lambda)
                // We compute G from f_plus/f_minus by dividing out lambda: G = f / lambda
                let energyPlus = 'N/A';
                let energyMinus = 'N/A';
                let lambdaPlusDisplay = 'N/A';
                let lambdaMinusDisplay = 'N/A';
                if (data && data.snapshot) {
                    const snapshot = data.snapshot;
                    const lambdaPlus = snapshot.lambda_plus;
                    const lambdaMinus = snapshot.lambda_minus;

                    // Format lambda values for display
                    if (lambdaPlus != null && isFinite(lambdaPlus)) {
                        lambdaPlusDisplay = lambdaPlus.toFixed(4);
                    }
                    if (lambdaMinus != null && isFinite(lambdaMinus)) {
                        lambdaMinusDisplay = lambdaMinus.toFixed(4);
                    }

                    if (data.f_plus && Array.isArray(data.f_plus) && data.f_plus.length > 0 && lambdaPlus != null && isFinite(lambdaPlus) && lambdaPlus !== 0) {
                        // Compute product tensor G_+ = f_+ / lambda_+, then energy = mean(G_+²)
                        const sumSq = data.f_plus.reduce((acc, val) => {
                            const gPlus = val / lambdaPlus; // Product tensor without lambda
                            return acc + gPlus * gPlus;
                        }, 0);
                        energyPlus = (sumSq / data.f_plus.length).toFixed(4);
                    }
                    if (data.f_minus && Array.isArray(data.f_minus) && data.f_minus.length > 0 && lambdaMinus != null && isFinite(lambdaMinus) && lambdaMinus !== 0) {
                        // Compute product tensor G_- = f_- / lambda_-, then energy = mean(G_-²)
                        const sumSq = data.f_minus.reduce((acc, val) => {
                            const gMinus = val / lambdaMinus; // Product tensor without lambda
                            return acc + gMinus * gMinus;
                        }, 0);
                        energyMinus = (sumSq / data.f_minus.length).toFixed(4);
                    }
                }

                // Create f+ label cell with scaling and energy
                const headerCellFPlus = document.createElement('div');
                headerCellFPlus.style.cssText = `padding: 8px; background: #f8f9fa; border: 1px solid #dee2e6; font-weight: bold; text-align: center; position: sticky; top: ${firstRowHeight}px; z-index: 10; font-size: 0.85rem; line-height: 1.3;`;
                headerCellFPlus.innerHTML = `
                    <div style="color: #3b82f6; font-weight: 600; margin-bottom: 2px;">f+</div>
                    <div style="font-size: 0.75rem; font-weight: normal; color: #495057;">
                        <div>Scaling: ${scalingPlus}</div>
                        <div>Energy: ${energyPlus}</div>
                        <div>λ+: ${lambdaPlusDisplay}</div>
                    </div>
                `;
                componentGrid.appendChild(headerCellFPlus);

                // Create f- label cell with scaling and energy
                const headerCellFMinus = document.createElement('div');
                headerCellFMinus.style.cssText = `padding: 8px; background: #f8f9fa; border: 1px solid #dee2e6; font-weight: bold; text-align: center; position: sticky; top: ${firstRowHeight}px; z-index: 10; font-size: 0.85rem; line-height: 1.3;`;
                headerCellFMinus.innerHTML = `
                    <div style="color: #ef4444; font-weight: 600; margin-bottom: 2px;">f-</div>
                    <div style="font-size: 0.75rem; font-weight: normal; color: #495057;">
                        <div>Scaling: ${scalingMinus}</div>
                        <div>Energy: ${energyMinus}</div>
                        <div>λ-: ${lambdaMinusDisplay}</div>
                    </div>
                `;
                componentGrid.appendChild(headerCellFMinus);
            }
        });

        // Collect lambda values for gradient coloring if toggle is enabled
        const useLambdaGradient = lambdaGradientToggle ? lambdaGradientToggle.checked : false;
        let lambdaPlusValues = [];
        let lambdaMinusValues = [];

        if (useLambdaGradient) {
            epochDataList.forEach(data => {
                if (data && data.snapshot) {
                    const lambdaPlus = data.snapshot.lambda_plus;
                    const lambdaMinus = data.snapshot.lambda_minus;
                    if (lambdaPlus != null && isFinite(lambdaPlus)) {
                        lambdaPlusValues.push(lambdaPlus);
                    }
                    if (lambdaMinus != null && isFinite(lambdaMinus)) {
                        lambdaMinusValues.push(lambdaMinus);
                    }
                }
            });
        }

        // Create color scales for gradient
        // Sort lambda values and create scales based on sorted positions
        let colorScalePlus = null;
        let colorScaleMinus = null;
        if (useLambdaGradient) {
            if (lambdaPlusValues.length > 0) {
                // Sort and get unique values
                const sortedPlus = [...new Set(lambdaPlusValues)].sort((a, b) => a - b);
                const minPlus = sortedPlus[0];
                const maxPlus = sortedPlus[sortedPlus.length - 1];
                if (minPlus !== maxPlus) {
                    colorScalePlus = d3.scaleSequential(d3.interpolateViridis)
                        .domain([minPlus, maxPlus]);
                } else {
                    colorScalePlus = () => d3.interpolateViridis(0.5);
                }
            }
            if (lambdaMinusValues.length > 0) {
                // Sort and get unique values
                const sortedMinus = [...new Set(lambdaMinusValues)].sort((a, b) => a - b);
                const minMinus = sortedMinus[0];
                const maxMinus = sortedMinus[sortedMinus.length - 1];
                if (minMinus !== maxMinus) {
                    colorScaleMinus = d3.scaleSequential(d3.interpolateInferno)
                        .domain([minMinus, maxMinus]);
                } else {
                    colorScaleMinus = () => d3.interpolateInferno(0.5);
                }
            }
        }

        // Create component rows
        for (let compIdx = 0; compIdx < maxComponents; compIdx++) {
            // Component label column (sticky)
            const compLabel = document.createElement('div');
            compLabel.style.cssText = 'padding: 8px; background: #f8f9fa; border: 1px solid #dee2e6; font-weight: bold; text-align: center; position: sticky; left: 0; z-index: 5;';
            compLabel.textContent = `Comp ${compIdx}`;
            componentGrid.appendChild(compLabel);

            // Create chart cells for each epoch (two or three cells per epoch depending on mode)
            visibleEpochsList.forEach((epoch, colIdx) => {
                const data = epochDataList[colIdx];
                const snapshot = data && data.snapshot ? data.snapshot : null;

                if (!snapshot || !snapshot.intervals || !snapshot.intervals[compIdx]) {
                    // No data - create empty cells (2 for f+/f-, 3 for backbone/tilt/tanh tilt)
                    const numCells = showBackboneTilt ? 3 : 2;
                    for (let i = 0; i < numCells; i++) {
                        const emptyCell = document.createElement('div');
                        emptyCell.style.cssText = 'width: 400px; height: 300px; border: 1px solid #dee2e6; padding: 0; overflow: hidden; box-sizing: border-box; background: #fff;';
                        emptyCell.innerHTML = '<p style="text-align: center; color: #999; line-height: 300px; margin: 0;">No data</p>';
                        componentGrid.appendChild(emptyCell);
                    }
                    return;
                }

                const ilist = snapshot.intervals[compIdx] || [];
                if (ilist.length === 0) {
                    const numCells = showBackboneTilt ? 3 : 2;
                    for (let i = 0; i < numCells; i++) {
                        const emptyCell = document.createElement('div');
                        emptyCell.style.cssText = 'width: 400px; height: 300px; border: 1px solid #dee2e6; padding: 0; overflow: hidden; box-sizing: border-box; background: #fff;';
                        emptyCell.innerHTML = '<p style="text-align: center; color: #999; line-height: 300px; margin: 0;">No data</p>';
                        componentGrid.appendChild(emptyCell);
                    }
                    return;
                }

                // Helper function to prepare intervals from values
                const prepareIntervals = (values) => {
                    if (!values || values.length === 0) return null;
                    const maxInt = 400;
                    const stride = Math.max(1, Math.ceil(values.length / maxInt));
                    const intervals = ilist
                        .map((ib, i) => [
                            UI.toFloat(ib[0]),
                            UI.toFloat(ib[1]),
                            UI.toFloat(values[i])
                        ])
                        .filter(v => v.length === 3 && v.every(x => isFinite(x)))
                        .filter((_, i) => i % stride === 0);
                    return intervals.length > 0 ? intervals : null;
                };

                let intervalsPlus, intervalsMinus, intervalsTanhTilt, intervalsBackboneSinhTilt;
                let tanhTiltDomain = null; // Will be set for tanh tilt charts
                if (showBackboneTilt) {
                    // Extract backbone and tilt values
                    const backbone = (snapshot.backbone_values || [])[compIdx] || [];
                    const tilt = (snapshot.tilt_values || [])[compIdx] || [];

                    intervalsPlus = prepareIntervals(backbone);
                    intervalsMinus = prepareIntervals(tilt);

                    // Compute tanh tilt: tanh(tilt) for each tilt value
                    // Always compute, even if tilt is empty (will result in null/empty intervals)
                    const tanhTilt = tilt.length > 0 ? tilt.map(d => Math.tanh(d)) : [];
                    intervalsTanhTilt = prepareIntervals(tanhTilt);

                    // Compute 2 × backbone × sinh(tilt): 2 * backbone[i] * sinh(tilt[i])
                    const backboneSinhTilt = (backbone.length > 0 && tilt.length > 0 && backbone.length === tilt.length)
                        ? backbone.map((b, i) => 2 * b * Math.sinh(tilt[i]))
                        : [];
                    intervalsBackboneSinhTilt = prepareIntervals(backboneSinhTilt);
                    
                    // Determine y-axis domain for tanh tilt: use smallest range that contains all values
                    if (intervalsTanhTilt && intervalsTanhTilt.length > 0) {
                        let tanhMin = Infinity;
                        let tanhMax = -Infinity;
                        intervalsTanhTilt.forEach(ib => {
                            const v = ib[2] == null ? 0 : +ib[2];
                            if (isFinite(v)) {
                                tanhMin = Math.min(tanhMin, v);
                                tanhMax = Math.max(tanhMax, v);
                            }
                        });
                        
                        if (isFinite(tanhMin) && isFinite(tanhMax)) {
                            // Check if values fit in [-0.5, 0.5]
                            if (tanhMin >= -0.5 && tanhMax <= 0.5) {
                                tanhTiltDomain = [-0.5, 0.5];
                            } else {
                                // Otherwise use full range [-1, 1]
                                tanhTiltDomain = [-1, 1];
                            }
                        } else {
                            // Default to full range if no valid values
                            tanhTiltDomain = [-1, 1];
                        }
                    } else {
                        // Default to full range if no data
                        tanhTiltDomain = [-1, 1];
                    }
                    
                    // Debug: verify tanh tilt is computed
                    if (compIdx === 0 && epoch === visibleEpochsList[0]) {
                        console.log('Tanh tilt computation:', {
                            tiltLength: tilt.length,
                            tanhTiltLength: tanhTilt.length,
                            intervalsTanhTilt: intervalsTanhTilt ? intervalsTanhTilt.length : null,
                            tanhTiltDomain: tanhTiltDomain
                        });
                    }
                } else {
                    // Compute f+ and f- components
                    const { f_plus, f_minus } = computeFComponents(snapshot, compIdx);
                    intervalsPlus = prepareIntervals(f_plus);
                    intervalsMinus = prepareIntervals(f_minus);
                    intervalsTanhTilt = null;
                    intervalsBackboneSinhTilt = null;
                }

                // Determine log scale from toggle (default to checked/log scale for f+/f-)
                const useLog = logScaleToggle ? logScaleToggle.checked : true;

                // Compute shared y-axis range only for f+/f- mode (not for backbone/tilt)
                let sharedYDomain = null;
                if (!showBackboneTilt && (intervalsPlus || intervalsMinus)) {
                    function sy(v) { return Math.sign(v) * Math.log10(1 + Math.abs(v)); }
                    let ymin = Infinity, ymax = -Infinity;
                    [intervalsPlus, intervalsMinus].forEach(intervals => {
                        if (intervals) {
                            intervals.forEach(ib => {
                                const v = ib[2] == null ? 0 : +ib[2];
                                const transformed = useLog ? sy(v) : v;
                                ymin = Math.min(ymin, transformed);
                                ymax = Math.max(ymax, transformed);
                            });
                        }
                    });
                    if (isFinite(ymin) && isFinite(ymax)) {
                        if (ymin === ymax) {
                            ymin -= 1;
                            ymax += 1;
                        }
                        // Apply .nice() once to the shared domain so both charts use the exact same range
                        const tempScale = d3.scaleLinear().domain([ymin, ymax]).nice();
                        const niceDomain = tempScale.domain();
                        sharedYDomain = [niceDomain[0], niceDomain[1]];
                    }
                }

                // Compute fixed tick count based on standard chart height (300px - margins = 248px inner height)
                const standardInnerH = 300 - 26 - 26; // height - top margin - bottom margin
                const fixedYTicksCount = Math.max(3, Math.min(6, Math.floor(standardInnerH / 40)));

                // Compute shared x-axis bounds - include ALL x-values from all datasets
                let allXValues = [];
                [intervalsPlus, intervalsMinus, intervalsTanhTilt, intervalsBackboneSinhTilt].forEach(intervals => {
                    if (intervals) {
                        intervals.forEach(ib => {
                            if (isFinite(ib[0])) allXValues.push(ib[0]);
                            if (isFinite(ib[1])) allXValues.push(ib[1]);
                        });
                    }
                });

                let lo = -1, hi = 1;
                if (allXValues.length > 0) {
                    lo = Math.min(...allXValues);
                    hi = Math.max(...allXValues);
                    if (!(isFinite(lo) && isFinite(hi)) || lo === hi) {
                        lo = -1;
                        hi = 1;
                    }
                }
                // Add small margin to ensure full range is visible
                const span = hi - lo;
                const margin = span > 0 ? 0.05 * span : 1.0;
                const bounds = { min: lo - margin, max: hi + margin };

                // Helper function to create a chart cell
                // chartType: 'backbone', 'tilt', 'tanh_tilt', 'backbone_sinh_tilt', 'f_plus', or 'f_minus'
                const createChartCell = (intervals, chartType) => {
                    const cell = document.createElement('div');
                    cell.className = 'chart-cell';
                    cell.style.cssText = 'width: 400px; height: 300px; border: 1px solid #dee2e6; padding: 0; overflow: hidden; box-sizing: border-box; background: #fff;';

                    if (!intervals || intervals.length === 0) {
                        cell.innerHTML = '<p style="text-align: center; color: #999; line-height: 300px; margin: 0;">No data</p>';
                        componentGrid.appendChild(cell);
                        return;
                    }

                    // Determine color based on mode
                    let lineColor;
                    let key;
                    if (showBackboneTilt) {
                        // Backbone/Tilt/Tanh Tilt/2×backbone×sinh(tilt) mode
                        if (chartType === 'backbone') {
                            lineColor = '#10b981';
                            key = `backbone_${compIdx}`;
                        } else if (chartType === 'tanh_tilt') {
                            lineColor = '#f59e0b';
                            key = `tanh_tilt_${compIdx}`;
                        } else if (chartType === 'backbone_sinh_tilt') {
                            lineColor = '#ec4899';
                            key = `backbone_sinh_tilt_${compIdx}`;
                        } else {
                            lineColor = '#8b5cf6';
                            key = `tilt_${compIdx}`;
                        }
                    } else {
                        // f+/f- mode: blue for f+, red for f-
                        if (chartType === 'f_plus') {
                            lineColor = '#3b82f6';
                            key = `f_plus_${compIdx}`;
                        } else {
                            lineColor = '#ef4444';
                            key = `f_minus_${compIdx}`;
                        }

                        // Apply lambda gradient if enabled
                        if (useLambdaGradient && snapshot) {
                            if (chartType === 'f_plus' && colorScalePlus) {
                                const lambdaPlus = snapshot.lambda_plus;
                                if (lambdaPlus != null && isFinite(lambdaPlus)) {
                                    lineColor = colorScalePlus(lambdaPlus);
                                }
                            } else if (chartType === 'f_minus' && colorScaleMinus) {
                                const lambdaMinus = snapshot.lambda_minus;
                                if (lambdaMinus != null && isFinite(lambdaMinus)) {
                                    lineColor = colorScaleMinus(lambdaMinus);
                                }
                            }
                        }
                    }

                    // Create chart directly in the cell
                    const comps = [{
                        intervals,
                        key: key,
                        color: lineColor
                    }];

                    // Ensure cell has explicit size and is in DOM before creating chart
                    cell.style.width = '400px';
                    cell.style.height = '300px';
                    componentGrid.appendChild(cell); // Append first so sizing works

                    // Determine y-axis domain
                    let yDomainToUse = null;
                    if (chartType === 'tanh_tilt') {
                        // Use dynamically determined domain: [-0.5, 0.5] or [-1, 1] depending on data range
                        yDomainToUse = tanhTiltDomain || [-1, 1];
                    } else if (!showBackboneTilt) {
                        // f+/f- mode uses shared domain
                        yDomainToUse = sharedYDomain;
                    }
                    // For backbone/tilt (non-tanh), yDomainToUse remains null (auto-scale)

                    // Create chart with y-axis domain
                    Charts.stepFunctions(cell, comps, bounds, {
                        title: '',
                        hoverable: false,
                        log: useLog,
                        yDomain: yDomainToUse,
                        yTicksCount: fixedYTicksCount  // Use fixed tick count for consistency
                    });

                    // Force SVG to be exactly the cell size and add lambda asymmetry annotation for tanh tilt
                    requestAnimationFrame(() => {
                        const svg = cell.querySelector('svg');
                        if (svg) {
                            // Set SVG to exact cell dimensions
                            svg.setAttribute('width', '400');
                            svg.setAttribute('height', '300');
                            svg.style.width = '400px';
                            svg.style.height = '300px';
                            svg.style.display = 'block';
                            
                            // Add lambda asymmetry horizontal line annotation for tanh tilt charts
                            if (chartType === 'tanh_tilt' && data && data.snapshot) {
                                const snapshot = data.snapshot;
                                const lambdaPlus = snapshot.lambda_plus;
                                const lambdaMinus = snapshot.lambda_minus;
                                const scalingPlus = data.scaling_plus != null && isFinite(data.scaling_plus) ? data.scaling_plus : 1.0;
                                const scalingMinus = data.scaling_minus != null && isFinite(data.scaling_minus) ? data.scaling_minus : 0.0;
                                
                                if (lambdaPlus != null && lambdaMinus != null && 
                                    isFinite(lambdaPlus) && isFinite(lambdaMinus)) {
                                    // Compute scaled lambda asymmetry
                                    const scaledLambdaPlus = scalingPlus * lambdaPlus;
                                    const scaledLambdaMinus = scalingMinus * lambdaMinus;
                                    const sum = scaledLambdaPlus + scaledLambdaMinus;
                                    
                                    if (sum !== 0 && isFinite(sum)) {
                                        const asymmetry = (scaledLambdaPlus - scaledLambdaMinus) / sum;
                                        
                                        // Clamp asymmetry to [-1, 1] range (tanh domain)
                                        const clampedAsymmetry = Math.max(-1, Math.min(1, asymmetry));
                                        // Plot at negative of asymmetry: y = -lambda_asymmetry
                                        const plotY = -clampedAsymmetry;
                                        
                                        // Get the inner chart group (where the plot is drawn)
                                        const d3Svg = d3.select(svg);
                                        const g = d3Svg.select('g'); // The main group with transform
                                        
                                        if (g.node()) {
                                            // Get chart dimensions from the chart
                                            const margin = { top: 26, right: 26, bottom: 26, left: 26 };
                                            const innerW = 400 - margin.left - margin.right;
                                            const innerH = 300 - margin.top - margin.bottom;
                                            
                                            // Use the same domain as the chart (tanhTiltDomain)
                                            const domain = tanhTiltDomain || [-1, 1];
                                            
                                            // Create y-scale for tanh tilt using the determined domain
                                            const yScale = d3.scaleLinear()
                                                .domain(domain)
                                                .range([innerH, 0]);
                                            
                                            // Draw horizontal dashed line at -asymmetry value
                                            g.append('line')
                                                .attr('x1', 0)
                                                .attr('x2', innerW)
                                                .attr('y1', yScale(plotY))
                                                .attr('y2', yScale(plotY))
                                                .attr('stroke', '#6b7280')
                                                .attr('stroke-width', 1.5)
                                                .attr('stroke-dasharray', '4,4')
                                                .attr('opacity', 0.7)
                                                .style('pointer-events', 'none');
                                            
                                            // Add text label at the right end of the line
                                            g.append('text')
                                                .attr('x', innerW - 4)
                                                .attr('y', yScale(plotY) - 4)
                                                .attr('fill', '#6b7280')
                                                .attr('font-size', '10px')
                                                .attr('text-anchor', 'end')
                                                .attr('font-weight', 500)
                                                .text(`-λ asym: ${(-asymmetry).toFixed(3)}`)
                                                .style('pointer-events', 'none');
                                        }
                                    }
                                }
                            }
                        }
                    });
                };

                if (showBackboneTilt) {
                    // Create first chart cell (backbone)
                    createChartCell(intervalsPlus, 'backbone');
                    // Create second chart cell (tilt)
                    createChartCell(intervalsMinus, 'tilt');
                    // Create third chart cell (tanh tilt) - always create, even if data is missing
                    createChartCell(intervalsTanhTilt, 'tanh_tilt');
                    // Create fourth chart cell (2×backbone×sinh(tilt))
                    createChartCell(intervalsBackboneSinhTilt, 'backbone_sinh_tilt');
                } else {
                    // Create first chart cell (f+)
                    createChartCell(intervalsPlus, 'f_plus');
                    // Create second chart cell (f-)
                    createChartCell(intervalsMinus, 'f_minus');
                }
            });
        }
    }

    // Event listeners
    componentRun.addEventListener('change', async () => {
        await loadEpochs();
    });

    if (logScaleToggle) {
        logScaleToggle.addEventListener('change', () => {
            render();
        });
    }

    if (lambdaGradientToggle) {
        lambdaGradientToggle.addEventListener('change', () => {
            render();
        });
    }

    if (backboneTiltToggle) {
        backboneTiltToggle.addEventListener('change', () => {
            render();
        });
    }

    // Initialize
    await loadRuns();
})();

