/* Reusable D3 chart helpers. All charts accept a container selector and data. */

const Charts = (() => {
    const theme = {
        bg: '#ffffff',
        grid: '#d6dee8',
        text: '#0f172a',
        axis: '#617186',
        palette: ['#C44E52', '#55A868', '#DD8452', '#8172B2', '#937860', '#DA8BC3', '#8C8C8C']
    };
    const api = {};
    api.__theme = theme;

    function clear(container) {
        d3.select(container).selectAll('*').remove();
    }

    // Simple reusable tooltip
    function getOrCreateTooltip() {
        let el = document.getElementById('d3-tooltip');
        if (!el) {
            el = document.createElement('div');
            el.id = 'd3-tooltip';
            el.style.position = 'absolute';
            el.style.pointerEvents = 'none';
            el.style.padding = '6px 8px';
            el.style.background = 'rgba(255,255,255,0.96)';
            el.style.border = '1px solid #cbd5e1';
            el.style.borderRadius = '6px';
            el.style.boxShadow = '0 4px 12px rgba(12,30,66,0.12)';
            el.style.fontSize = '12px';
            el.style.color = '#0f172a';
            el.style.display = 'none';
            document.body.appendChild(el);
        }
        return el;
    }

    function sized(container, opts = {}) {
        const enforceSquare = opts.enforceSquare === true;
        const node = d3.select(container).node();
        const rect = node ? node.getBoundingClientRect() : { width: 0 };
        const fallbackW = (node && node.parentElement) ? node.parentElement.getBoundingClientRect().width : 0;
        let width = rect.width || fallbackW || Math.max(320, Math.min(window.innerWidth || 800, 960));
        // Apply caller-provided caps if present
        if (opts && Number.isFinite(opts.maxWidth)) {
            width = Math.min(width, Number(opts.maxWidth));
        }
        // Target ~3x3 grid visible: base height responsive to viewport height
        const vh = Math.max(window.innerHeight || 800, 600);
        let height;
        if (enforceSquare) {
            height = width;
        } else {
            height = Math.max(140, Math.min(vh / 3 - 24, 260));
        }
        // Respect maxHeight if provided
        if (opts && Number.isFinite(opts.maxHeight)) {
            height = Math.min(height, Number(opts.maxHeight));
        }
        // Increase left margin to keep y tick labels inside the chart
        const margin = { top: 26, right: 18, bottom: 26, left: 56 };
        return { width, height, margin, innerW: width - margin.left - margin.right, innerH: height - margin.top - margin.bottom };
    }

    // Creates a fullscreen modal with greyed background and renders content via callback
    function openModalZoom(sourceContainer, renderCb, currentDomain) {
        // Remove any existing modal
        const existing = document.getElementById('chart-modal-overlay');
        if (existing) existing.remove();

        const overlay = document.createElement('div');
        overlay.id = 'chart-modal-overlay';
        overlay.style.position = 'fixed';
        overlay.style.inset = '0';
        overlay.style.background = 'rgba(15, 23, 42, 0.55)';
        overlay.style.display = 'flex';
        overlay.style.alignItems = 'center';
        overlay.style.justifyContent = 'center';
        overlay.style.zIndex = '10000';

        const modal = document.createElement('div');
        modal.className = 'chart-modal';
        modal.style.background = '#ffffff';
        modal.style.border = '1px solid #e2e8f0';
        modal.style.borderRadius = '10px';
        modal.style.boxShadow = '0 20px 60px rgba(0,0,0,0.25)';
        modal.style.maxWidth = '92vw';
        modal.style.maxHeight = '86vh';
        modal.style.width = 'min(1100px, 92vw)';
        modal.style.height = 'min(700px, 86vh)';
        modal.style.display = 'flex';
        modal.style.flexDirection = 'column';

        const header = document.createElement('div');
        header.style.display = 'flex';
        header.style.alignItems = 'center';
        header.style.justifyContent = 'space-between';
        header.style.padding = '10px 12px';
        header.style.borderBottom = '1px solid #e2e8f0';
        const title = document.createElement('div');
        title.textContent = 'Chart Zoom';
        title.style.fontWeight = '600';
        title.style.color = '#0f172a';
        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'Close';
        closeBtn.style.background = '#1f2937';
        closeBtn.style.color = '#fff';
        closeBtn.style.border = 'none';
        closeBtn.style.borderRadius = '6px';
        closeBtn.style.padding = '6px 10px';
        closeBtn.style.cursor = 'pointer';
        closeBtn.onclick = () => overlay.remove();
        header.appendChild(title);
        header.appendChild(closeBtn);

        const body = document.createElement('div');
        body.style.flex = '1 1 auto';
        body.style.padding = '10px';
        body.style.overflow = 'auto';

        const target = document.createElement('div');
        target.className = 'chart';
        target.style.minHeight = '320px';
        body.appendChild(target);

        modal.appendChild(header);
        modal.appendChild(body);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);

        function onKey(ev) {
            if (ev.key === 'Escape') {
                overlay.remove();
                document.removeEventListener('keydown', onKey);
            }
        }
        document.addEventListener('keydown', onKey);
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.remove();
        });

        // Render content
        if (typeof renderCb === 'function') {
            renderCb(target, currentDomain);
        }
    }

    function lineChart(container, series, { xKey, yKey, title, color = theme.palette[0], log = false }) {
        clear(container);
        const { width, height, margin, innerW, innerH } = sized(container);
        const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleLinear()
            .domain(d3.extent(series, d => +d[xKey] ?? 0)).nice()
            .range([0, innerW]);
        const useLog = log === true;
        function sy(v) { return Math.sign(v) * Math.log10(1 + Math.abs(v)); }
        function invSy(t) { return Math.sign(t) * (Math.pow(10, Math.abs(t)) - 1); }
        const yDomainSrc = series.map(d => +d[yKey] ?? 0);
        const yTrans = useLog ? yDomainSrc.map(sy) : yDomainSrc;
        let yExtent = d3.extent(yTrans);
        if (yExtent[0] === yExtent[1]) { yExtent = [yExtent[0] - 1, yExtent[1] + 1]; }
        const y = d3.scaleLinear()
            .domain(yExtent).nice()
            .range([innerH, 0]);

        const idealTicks = Math.max(3, Math.min(6, Math.floor(innerW / 80)));
        const yTicks = Math.max(3, Math.min(6, Math.floor(innerH / 40)));
        const xAxis = g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(idealTicks));
        const yTickFormat = (t) => {
            if (!useLog) return d3.format("~g")(t);
            const v = invSy(t);
            const abs = Math.abs(v);
            if (abs >= 1) return d3.format("~g")(v);
            return d3.format(".3f")(v);
        };
        const yAxis = g.append('g').call(d3.axisLeft(y).ticks(yTicks).tickFormat(yTickFormat));
        xAxis.selectAll('text').attr('fill', theme.axis);
        yAxis.selectAll('text').attr('fill', theme.axis);
        xAxis.selectAll('path,line').attr('stroke', theme.grid);
        yAxis.selectAll('path,line').attr('stroke', theme.grid);

        const line = d3.line().x(d => x(+d[xKey] ?? 0)).y(d => y(useLog ? sy(+d[yKey] ?? 0) : (+d[yKey] ?? 0))).curve(d3.curveMonotoneX);
        g.append('path')
            .datum(series)
            .attr('fill', 'none')
            .attr('stroke', color)
            .attr('stroke-width', 1.6)
            .attr('d', line);

        // Hover: nearest point, focus marker, tooltip
        const focus = g.append('circle').attr('r', 3.2).attr('fill', color).style('display', 'none');
        const tooltip = getOrCreateTooltip();
        const overlay = g.append('rect').attr('fill', 'transparent').attr('width', innerW).attr('height', innerH).style('cursor', 'crosshair');
        const bisectX = d3.bisector(d => +d[xKey]).center;
        overlay.on('mousemove', function (event) {
            const [mx, my] = d3.pointer(event, this);
            const xVal = x.invert(mx);
            const idx = Math.max(0, Math.min(series.length - 1, bisectX(series, xVal)));
            const d0 = series[idx];
            const px = x(+d0[xKey] ?? 0);
            const py = y(useLog ? sy(+d0[yKey] ?? 0) : (+d0[yKey] ?? 0));
            focus.style('display', null).attr('cx', px).attr('cy', py);
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 12) + 'px';
            tooltip.style.top = (event.pageY + 12) + 'px';
            tooltip.innerHTML = `<b>${title || ''}</b><div>${xKey}: ${(+d0[xKey]).toLocaleString()}</div><div>${yKey}: ${(+d0[yKey]).toLocaleString()}</div>`;
        }).on('mouseleave', function () {
            focus.style('display', 'none');
            tooltip.style.display = 'none';
        });

        if (title) svg.append('text').attr('x', margin.left + 6).attr('y', 18).attr('fill', theme.text).attr('font-weight', 600).text(title);
    }

    function barChart(container, series, { xKey, yKey, title, color = theme.palette[0] }) {
        clear(container);
        const { width, height, margin, innerW, innerH } = sized(container);
        const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleBand().domain(series.map(d => d[xKey])).range([0, innerW]).padding(0.15);
        const y = d3.scaleLinear().domain([0, d3.max(series, d => +d[yKey] ?? 0) || 1]).nice().range([innerH, 0]);

        const idealTicks = Math.max(3, Math.min(6, Math.floor(innerW / 80)));
        const yTicks = Math.max(3, Math.min(6, Math.floor(innerH / 40)));
        const xAxis = g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(idealTicks));
        const yAxis = g.append('g').call(d3.axisLeft(y).ticks(yTicks));
        xAxis.selectAll('text').attr('fill', theme.axis);
        yAxis.selectAll('text').attr('fill', theme.axis);
        xAxis.selectAll('path,line').attr('stroke', theme.grid);
        yAxis.selectAll('path,line').attr('stroke', theme.grid);

        g.selectAll('rect').data(series).join('rect')
            .attr('x', d => x(d[xKey]))
            .attr('y', d => y(+d[yKey]))
            .attr('width', x.bandwidth())
            .attr('height', d => innerH - y(+d[yKey]))
            .attr('fill', color)
            .on('mousemove', function (event, d) {
                const tooltip = getOrCreateTooltip();
                d3.select(this).attr('opacity', 0.85);
                tooltip.style.display = 'block';
                tooltip.style.left = (event.pageX + 12) + 'px';
                tooltip.style.top = (event.pageY + 12) + 'px';
                tooltip.innerHTML = `<b>${title || ''}</b><div>${xKey}: ${d[xKey]}</div><div>${yKey}: ${(+d[yKey]).toLocaleString()}</div>`;
            })
            .on('mouseleave', function () {
                d3.select(this).attr('opacity', 1.0);
                const tooltip = getOrCreateTooltip();
                tooltip.style.display = 'none';
            });

        if (title) svg.append('text').attr('x', margin.left + 6).attr('y', 18).attr('fill', theme.text).attr('font-weight', 600).text(title);
    }

    function stackedBars(container, series, xKey, yKeys, colors, title) {
        clear(container);
        const { width, height, margin, innerW, innerH } = sized(container);
        const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleBand().domain(series.map(d => d[xKey])).range([0, innerW]).padding(0.15);
        const y = d3.scaleLinear()
            .domain([0, d3.max(series, d => yKeys.reduce((s, k) => s + (+d[k] || 0), 0)) || 1])
            .nice().range([innerH, 0]);

        const xAxis = g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x));
        const yAxis = g.append('g').call(d3.axisLeft(y).ticks(6));
        xAxis.selectAll('text').attr('fill', theme.axis);
        yAxis.selectAll('text').attr('fill', theme.axis);
        xAxis.selectAll('path,line').attr('stroke', theme.grid);
        yAxis.selectAll('path,line').attr('stroke', theme.grid);

        let offset = series.map(() => 0);
        yKeys.forEach((k, idx) => {
            const color = colors[idx % colors.length];
            g.selectAll(`rect.layer-${idx}`)
                .data(series)
                .join('rect')
                .attr('class', `layer-${idx}`)
                .attr('x', d => x(d[xKey]))
                .attr('y', (d, i) => y(offset[i] + (+d[k] || 0)))
                .attr('width', x.bandwidth())
                .attr('height', (d, i) => y(offset[i]) - y(offset[i] + (+d[k] || 0)))
                .attr('fill', color)
                .on('mousemove', function (event, d) {
                    const tooltip = getOrCreateTooltip();
                    d3.select(this).attr('opacity', 0.85);
                    const val = +d[k] || 0;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.pageX + 12) + 'px';
                    tooltip.style.top = (event.pageY + 12) + 'px';
                    tooltip.innerHTML = `<b>${title || ''}</b><div>${xKey}: ${d[xKey]}</div><div>${k}: ${val.toLocaleString()}</div>`;
                })
                .on('mouseleave', function () {
                    d3.select(this).attr('opacity', 1.0);
                    const tooltip = getOrCreateTooltip();
                    tooltip.style.display = 'none';
                });
            offset = offset.map((v, i) => v + (+series[i][k] || 0));
        });

        if (title) svg.append('text').attr('x', margin.left + 6).attr('y', 18).attr('fill', theme.text).attr('font-weight', 600).text(title);
    }

    function multiLineChart(container, groupedSeries, { xKey, yKey, title, log = false }) {
        // groupedSeries: [{ name, color, data: [{xKey, yKey}, ...] }]
        clear(container);
        const { width, height, margin, innerW, innerH } = sized(container);
        const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
        const all = groupedSeries.flatMap(s => s.data);
        const x = d3.scaleLinear()
            .domain(d3.extent(all, d => +d[xKey] ?? 0)).nice()
            .range([0, innerW]);
        const useLog = log === true;
        function sy(v) { return Math.sign(v) * Math.log10(1 + Math.abs(v)); }
        function invSy(t) { return Math.sign(t) * (Math.pow(10, Math.abs(t)) - 1); }
        const yDomainSrc = all.map(d => +d[yKey] ?? 0);
        const yTrans = useLog ? yDomainSrc.map(sy) : yDomainSrc;
        let yExtent = d3.extent(yTrans);
        if (yExtent[0] === yExtent[1]) { yExtent = [yExtent[0] - 1, yExtent[1] + 1]; }
        const y = d3.scaleLinear()
            .domain(yExtent).nice()
            .range([innerH, 0]);
        const idealTicks = Math.max(3, Math.min(6, Math.floor(innerW / 80)));
        const yTicks = Math.max(3, Math.min(6, Math.floor(innerH / 40)));
        const xAxis = g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(idealTicks));
        const yTickFormat = (t) => {
            if (!useLog) return d3.format("~g")(t);
            const v = invSy(t);
            const abs = Math.abs(v);
            if (abs >= 1) return d3.format("~g")(v);
            return d3.format(".3f")(v);
        };
        const yAxis = g.append('g').call(d3.axisLeft(y).ticks(yTicks).tickFormat(yTickFormat));
        xAxis.selectAll('text').attr('fill', theme.axis);
        yAxis.selectAll('text').attr('fill', theme.axis);
        xAxis.selectAll('path,line').attr('stroke', theme.grid);
        yAxis.selectAll('path,line').attr('stroke', theme.grid);

        const line = d3.line()
            .x(d => x(+d[xKey] ?? 0))
            .y(d => y(useLog ? sy(+d[yKey] ?? 0) : (+d[yKey] ?? 0)))
            .curve(d3.curveMonotoneX);

        groupedSeries.forEach(s => {
            const color = s.color || theme.palette[0];
            const sortedData = [...s.data].sort((a, b) => +a[xKey] - +b[xKey]);
            // Sanitize name for CSS class selector (replace spaces and special chars)
            const safeName = s.name.replace(/[^a-zA-Z0-9]/g, '_');
            g.append('path')
                .datum(sortedData)
                .attr('fill', 'none')
                .attr('stroke', color)
                .attr('stroke-width', 1.6)
                .attr('d', line)
                .attr('opacity', 0.85)
                .attr('data-series-name', s.name)
                .style('cursor', 'pointer')
                .on('mouseenter', function (event) {
                    d3.select(this).attr('stroke-width', 2.4);
                    const tooltip = getOrCreateTooltip();
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.pageX + 12) + 'px';
                    tooltip.style.top = (event.pageY + 12) + 'px';
                    tooltip.innerHTML = `<b>${s.name}</b>`;
                })
                .on('mousemove', function (event) {
                    const tooltip = getOrCreateTooltip();
                    tooltip.style.left = (event.pageX + 12) + 'px';
                    tooltip.style.top = (event.pageY + 12) + 'px';
                })
                .on('mouseleave', function () {
                    d3.select(this).attr('stroke-width', 1.6);
                    const tooltip = getOrCreateTooltip();
                    tooltip.style.display = 'none';
                });

            // Add hover points
            g.selectAll(`circle.${safeName}`)
                .data(sortedData)
                .join('circle')
                .attr('class', safeName)
                .attr('data-series-name', s.name)
                .attr('cx', d => x(+d[xKey] ?? 0))
                .attr('cy', d => y(useLog ? sy(+d[yKey] ?? 0) : (+d[yKey] ?? 0)))
                .attr('r', 2.4)
                .attr('fill', color)
                .attr('opacity', 0.85)
                .on('mousemove', function (event, d) {
                    const tooltip = getOrCreateTooltip();
                    d3.select(this).attr('r', 3.2);
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.pageX + 12) + 'px';
                    tooltip.style.top = (event.pageY + 12) + 'px';
                    tooltip.innerHTML = `<b>${title || ''}</b><div>${s.name}</div><div>${xKey}: ${(+d[xKey]).toLocaleString()}</div><div>${yKey}: ${(+d[yKey]).toLocaleString()}</div>`;
                })
                .on('mouseleave', function () {
                    d3.select(this).attr('r', 2.4);
                    const tooltip = getOrCreateTooltip();
                    tooltip.style.display = 'none';
                });
        });

        if (title) svg.append('text').attr('x', margin.left + 6).attr('y', 18).attr('fill', theme.text).attr('font-weight', 600).text(title);
    }

    function scatter(container, groupedSeries, opts = {}) {
        // groupedSeries: [{ name, color, data: [{xKey, yKey}, ...] }]
        // opts: { xKey, yKey, title, log, square, maxWidth, maxHeight }
        const { xKey, yKey, title, log = false, square = false, maxWidth = null, maxHeight = null } = opts || {};
        clear(container);
        const { width, height, margin, innerW, innerH } = sized(container, { enforceSquare: square, maxWidth, maxHeight });
        const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
        const all = groupedSeries.flatMap(s => s.data);
        const x = d3.scaleLinear()
            .domain(d3.extent(all, d => +d[xKey] ?? 0)).nice()
            .range([0, innerW]);
        const useLog = log === true;
        function sy(v) { return Math.sign(v) * Math.log10(1 + Math.abs(v)); }
        function invSy(t) { return Math.sign(t) * (Math.pow(10, Math.abs(t)) - 1); }
        const yDomainSrc = all.map(d => +d[yKey] ?? 0);
        const yTrans = useLog ? yDomainSrc.map(sy) : yDomainSrc;
        let yExtent = d3.extent(yTrans);
        if (yExtent[0] === yExtent[1]) { yExtent = [yExtent[0] - 1, yExtent[1] + 1]; }
        const y = d3.scaleLinear()
            .domain(yExtent).nice()
            .range([innerH, 0]);
        const idealTicks = Math.max(3, Math.min(6, Math.floor(innerW / 80)));
        const yTicks = Math.max(3, Math.min(6, Math.floor(innerH / 40)));
        const xAxis = g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(idealTicks));
        const yTickFormat = (t) => {
            if (!useLog) return d3.format("~g")(t);
            const v = invSy(t);
            const abs = Math.abs(v);
            if (abs >= 1) return d3.format("~g")(v);
            return d3.format(".3f")(v);
        };
        const yAxis = g.append('g').call(d3.axisLeft(y).ticks(yTicks).tickFormat(yTickFormat));
        xAxis.selectAll('text').attr('fill', theme.axis);
        yAxis.selectAll('text').attr('fill', theme.axis);
        xAxis.selectAll('path,line').attr('stroke', theme.grid);
        yAxis.selectAll('path,line').attr('stroke', theme.grid);

        groupedSeries.forEach(s => {
            const color = s.color || theme.palette[0];
            g.selectAll(`circle.${s.name}`)
                .data(s.data)
                .join('circle')
                .attr('class', s.name)
                .attr('cx', d => x(+d[xKey]))
                .attr('cy', d => y(useLog ? sy(+d[yKey]) : +d[yKey]))
                .attr('r', s.r || 2.4)
                .attr('fill', color)
                .attr('opacity', 0.85)
                .on('mousemove', function (event, d) {
                    const tooltip = getOrCreateTooltip();
                    d3.select(this).attr('r', 3.2);
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.pageX + 12) + 'px';
                    tooltip.style.top = (event.pageY + 12) + 'px';
                    tooltip.innerHTML = `<b>${title || ''}</b><div>${xKey}: ${(+d[xKey]).toLocaleString()}</div><div>${yKey}: ${(+d[yKey]).toLocaleString()}</div>`;
                })
                .on('mouseleave', function () {
                    d3.select(this).attr('r', 2.4);
                    const tooltip = getOrCreateTooltip();
                    tooltip.style.display = 'none';
                });
        });

        // Ensure the 'best' series (if present) is rendered on top of other points
        try {
            g.selectAll('circle.best').raise();
        } catch (e) {
            // raise() may not exist on older d3 versions; ignore if unavailable
        }

        if (title) svg.append('text').attr('x', margin.left + 6).attr('y', 18).attr('fill', theme.text).attr('font-weight', 600).text(title);
    }

    function scatterWithSelection(container, groupedSeries, { xKey, yKey, title, log = false, onSelectionChange = null }) {
        // groupedSeries: [{ name, color, data: [{xKey, yKey}, ...] }]
        clear(container);
        const { width, height, margin, innerW, innerH } = sized(container);
        const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
        const all = groupedSeries.flatMap(s => s.data);
        const x = d3.scaleLinear()
            .domain(d3.extent(all, d => +d[xKey] ?? 0)).nice()
            .range([0, innerW]);
        const useLog = log === true;
        function sy(v) { return Math.sign(v) * Math.log10(1 + Math.abs(v)); }
        function invSy(t) { return Math.sign(t) * (Math.pow(10, Math.abs(t)) - 1); }
        const yDomainSrc = all.map(d => +d[yKey] ?? 0);
        const yTrans = useLog ? yDomainSrc.map(sy) : yDomainSrc;
        let yExtent = d3.extent(yTrans);
        if (yExtent[0] === yExtent[1]) { yExtent = [yExtent[0] - 1, yExtent[1] + 1]; }
        const y = d3.scaleLinear()
            .domain(yExtent).nice()
            .range([innerH, 0]);
        const idealTicks = Math.max(3, Math.min(6, Math.floor(innerW / 80)));
        const yTicks = Math.max(3, Math.min(6, Math.floor(innerH / 40)));
        const xAxis = g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(idealTicks));
        const yTickFormat = (t) => {
            if (!useLog) return d3.format("~g")(t);
            const v = invSy(t);
            const abs = Math.abs(v);
            if (abs >= 1) return d3.format("~g")(v);
            return d3.format(".3f")(v);
        };
        const yAxis = g.append('g').call(d3.axisLeft(y).ticks(yTicks).tickFormat(yTickFormat));
        xAxis.selectAll('text').attr('fill', theme.axis);
        yAxis.selectAll('text').attr('fill', theme.axis);
        xAxis.selectAll('path,line').attr('stroke', theme.grid);
        yAxis.selectAll('path,line').attr('stroke', theme.grid);

        // Prefer true/original epoch (used when x is jittered for display)
        const getTrueEpoch = (d) => {
            if (!d) return d;
            // Common places we may store the original epoch
            const td = d.tooltipData || {};
            return td.Epoch ?? td.epoch ?? d.originalEpoch ?? d.original_epoch ?? d.trueEpoch ?? d[xKey];
        };

        // Create selection rectangle
        const selectionRect = g.append('rect')
            .attr('class', 'selection-rect')
            .attr('fill', 'rgba(59, 130, 246, 0.15)')
            .attr('stroke', 'rgba(59, 130, 246, 0.9)')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '4,4')
            .style('display', 'none')
            .style('pointer-events', 'none');

        // Create selection info container below the chart
        const selectionInfo = d3.select(container).append('div')
            .attr('class', 'selection-info')
            .style('margin-top', '12px')
            .style('padding', '12px')
            .style('background', '#f8fafc')
            .style('border', '1px solid #e2e8f0')
            .style('border-radius', '6px')
            .style('font-size', '13px')
            .style('min-height', '60px')
            .style('display', 'none');

        // Add selection info header
        const selectionHeader = selectionInfo.append('div')
            .style('font-weight', '600')
            .style('margin-bottom', '8px')
            .style('color', '#374151');

        selectionHeader.text('Selected Points');

        // Add selection info content
        const selectionContent = selectionInfo.append('div')
            .attr('class', 'selection-content')
            .style('color', '#6b7280');

        let isSelecting = false;
        let startPoint = null;

        // Mouse event handlers for rectangle selection
        const overlay = g.append('rect')
            .attr('fill', 'transparent')
            .attr('width', innerW)
            .attr('height', innerH)
            .style('cursor', 'crosshair');

        overlay.on('mousedown', function (event) {
            if (event.button !== 0) return; // Only left mouse button
            isSelecting = true;
            const [mx, my] = d3.pointer(event, this);
            startPoint = { x: mx, y: my };
            selectionRect.style('display', null)
                .attr('x', mx)
                .attr('y', my)
                .attr('width', 0)
                .attr('height', 0);
        });

        overlay.on('mousemove', function (event) {
            if (!isSelecting || !startPoint) return;
            const [mx, my] = d3.pointer(event, this);
            const x1 = Math.min(startPoint.x, mx);
            const y1 = Math.min(startPoint.y, my);
            const x2 = Math.max(startPoint.x, mx);
            const y2 = Math.max(startPoint.y, my);

            selectionRect
                .attr('x', x1)
                .attr('y', y1)
                .attr('width', x2 - x1)
                .attr('height', y2 - y1);
        });

        overlay.on('mouseup', function (event) {
            if (!isSelecting || !startPoint) return;
            isSelecting = false;

            const [mx, my] = d3.pointer(event, this);
            const x1 = Math.min(startPoint.x, mx);
            const y1 = Math.min(startPoint.y, my);
            const x2 = Math.max(startPoint.x, mx);
            const y2 = Math.max(startPoint.y, my);

            // Convert pixel coordinates to data coordinates
            const xMin = x.invert(x1);
            const xMax = x.invert(x2);
            const yMin = y.invert(y2); // Note: y-axis is inverted
            const yMax = y.invert(y1);

            // Find selected points
            const selectedPoints = [];
            groupedSeries.forEach(series => {
                series.data.forEach(point => {
                    const px = +point[xKey];
                    const py = +point[yKey];
                    const pyTransformed = useLog ? sy(py) : py;

                    if (px >= xMin && px <= xMax && pyTransformed >= yMin && pyTransformed <= yMax) {
                        selectedPoints.push({
                            ...point,
                            seriesName: series.name,
                            seriesColor: series.color
                        });
                    }
                });
            });

            // Update selection info
            if (selectedPoints.length > 0) {
                selectionInfo.style('display', 'block');
                selectionHeader.text(`Selected Points (${selectedPoints.length})`);
                const content = selectedPoints.map(point => {
                    const treeId = point.tooltipData?.['Tree ID'] || point.tree_id || 'N/A';
                    const epoch = getTrueEpoch(point);
                    const error = +point[yKey];
                    const errText = Number.isFinite(error) ? error.toFixed(6) : String(point[yKey] ?? '');
                    return `<div style="margin-bottom: 4px;">
                                <span style="color: ${point.seriesColor}">●</span> 
                                Product ${treeId}, Stage ${epoch}, Error: ${errText}
                            </div>`;
                }).join('');
                selectionContent.html(content);

                // Call callback if provided
                if (onSelectionChange) {
                    onSelectionChange(selectedPoints);
                }
            } else {
                selectionInfo.style('display', 'none');
                selectionHeader.text('Selected Points');

                // Clear any navigation messages
                selectionInfo.select('.nav-message').remove();
            }

            // Hide selection rectangle
            selectionRect.style('display', 'none');
            startPoint = null;
        });

        // Draw the scatter points
        groupedSeries.forEach(s => {
            const color = s.color || theme.palette[0];
            g.selectAll(`circle.${s.name}`)
                .data(s.data)
                .join('circle')
                .attr('class', s.name)
                .attr('cx', d => x(+d[xKey]))
                .attr('cy', d => y(useLog ? sy(+d[yKey]) : +d[yKey]))
                .attr('r', 2.4)
                .attr('fill', color)
                .attr('opacity', 0.85)
                .on('mousemove', function (event, d) {
                    const tooltip = getOrCreateTooltip();
                    d3.select(this).attr('r', 3.2);
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.pageX + 12) + 'px';
                    tooltip.style.top = (event.pageY + 12) + 'px';
                    const epochVal = getTrueEpoch(d);
                    const errVal = +d[yKey];
                    const treeId = d.tooltipData?.['Tree ID'] ?? d.tree_id;
                    const parts = [
                        `<b>${title || ''}</b>`,
                        `<div>Stage: ${Number.isFinite(+epochVal) ? (+epochVal).toLocaleString() : String(epochVal ?? '')}</div>`,
                        `<div>Error: ${Number.isFinite(errVal) ? errVal.toLocaleString() : String(d[yKey] ?? '')}</div>`
                    ];
                    if (treeId != null) parts.push(`<div>Product: ${treeId}</div>`);
                    tooltip.innerHTML = parts.join('');
                })
                .on('mouseleave', function () {
                    d3.select(this).attr('r', 2.4);
                    const tooltip = getOrCreateTooltip();
                    tooltip.style.display = 'none';
                });
        });

        // Ensure the 'best' series (if present) is rendered on top of other points
        try {
            g.selectAll('circle.best').raise();
        } catch (e) {
            // raise() may not exist on older d3 versions; ignore if unavailable
        }

        if (title) svg.append('text').attr('x', margin.left + 6).attr('y', 18).attr('fill', theme.text).attr('font-weight', 600).text(title);
    }

    function stepFunctions(container, comps, bounds, options = {}) {
        // comps: [{ intervals: [[a,b,val],...], color, title }]
        clear(container);
        const { width, height, margin, innerW, innerH } = sized(container);
        const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
        const [lo, hi] = [bounds.min, bounds.max];
        const xDomain = (options && Array.isArray(options.xDomain) && options.xDomain.length === 2)
            ? options.xDomain
            : [lo, hi];
        const x = d3.scaleLinear().domain(xDomain).range([0, innerW]);
        // y-scale with optional pseudo-log transform: sign(y)*log10(1+|y|)
        const useLog = options && options.log === true;
        function sy(v) { return Math.sign(v) * Math.log10(1 + Math.abs(v)); }
        function invSy(t) { return Math.sign(t) * (Math.pow(10, Math.abs(t)) - 1); }
        const eps = 1e-12;
        // Compute transformed value arrays per comp, with baseline 1 for all-zero comps
        const transformedComps = comps.map(c => {
            const allZero = (c.intervals || []).every(iv => Math.abs(iv[2] || 0) < eps);
            const baseVal = useLog ? sy(1) : 1;
            const arr = (c.intervals || []).map(iv => {
                const v = iv[2] == null ? 0 : +iv[2];
                return allZero ? baseVal : (useLog ? sy(v) : v);
            });
            return { key: c.key, color: c.color, intervals: c.intervals || [], transformedVals: arr, split_value: c.split_value, split_label: c.split_label };
        });
        let ymin = Infinity, ymax = -Infinity;
        let useProvidedDomain = false;
        let sharedTickValues = null;
        // If yDomain is provided, use a fixed tick count (based on standard chart height of 300px)
        // to ensure both charts generate identical ticks regardless of slight measurement differences
        const standardInnerH = 300 - 26 - 26; // height - top margin - bottom margin
        const fixedYTicks = options && options.yTicksCount
            ? options.yTicksCount
            : Math.max(3, Math.min(6, Math.floor(standardInnerH / 40)));
        const yTicks = Math.max(3, Math.min(6, Math.floor(innerH / 40)));
        // If yDomain is provided, use it; otherwise compute from data
        if (options && Array.isArray(options.yDomain) && options.yDomain.length === 2) {
            ymin = options.yDomain[0];
            ymax = options.yDomain[1];
            useProvidedDomain = true;
            // Compute tick values once from the shared domain to ensure both charts use identical ticks
            // Use a temporary scale with the same domain to generate consistent tick values
            const tempScale = d3.scaleLinear().domain([ymin, ymax]);
            sharedTickValues = tempScale.ticks(fixedYTicks);
        } else {
            transformedComps.forEach(tc => {
                tc.transformedVals.forEach(v => { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); });
            });
            if (!isFinite(ymin) || !isFinite(ymax)) { ymin = -1; ymax = 1; }
            if (ymin === ymax) { ymin -= 1; ymax += 1; }
        }
        // Only apply .nice() if we computed the domain from data; if provided, use it as-is
        const y = useProvidedDomain
            ? d3.scaleLinear().domain([ymin, ymax]).range([innerH, 0])
            : d3.scaleLinear().domain([ymin, ymax]).nice().range([innerH, 0]);
        const idealTicks = Math.max(3, Math.min(6, Math.floor(innerW / 80)));
        const xAxis = g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(idealTicks));
        const yTickFormat = (t) => {
            if (!useLog) return d3.format("~g")(t);
            const v = invSy(t);
            // Compact formatting for original domain
            const abs = Math.abs(v);
            if (abs >= 1) return d3.format("~g")(v);
            return d3.format(".3f")(v);
        };
        // Use explicit tick values if provided (for shared y-axis), otherwise let d3 generate them
        const yAxisGenerator = useProvidedDomain && sharedTickValues && sharedTickValues.length > 0
            ? d3.axisLeft(y).tickValues(sharedTickValues).tickFormat(yTickFormat)
            : d3.axisLeft(y).ticks(yTicks).tickFormat(yTickFormat);
        const yAxis = g.append('g').call(yAxisGenerator);
        xAxis.selectAll('text').attr('fill', theme.axis);
        yAxis.selectAll('text').attr('fill', theme.axis);
        xAxis.selectAll('path,line').attr('stroke', theme.grid);
        yAxis.selectAll('path,line').attr('stroke', theme.grid);

        // Clip path to keep lines within axes area
        const clipId = `clip-${Math.random().toString(36).slice(2)}`;
        const defs = svg.append('defs');
        defs.append('clipPath').attr('id', clipId)
            .append('rect').attr('x', 0).attr('y', 0).attr('width', innerW).attr('height', innerH);

        transformedComps.forEach((comp, idx) => {
            const color = comp.color || theme.palette[idx % theme.palette.length];
            const data = comp.intervals || [];
            const points = [];
            data.forEach(([a, b, val], i) => {
                const left = Number.isFinite(a) ? a : lo;
                const right = Number.isFinite(b) ? b : hi;
                const vy = comp.transformedVals[i];
                points.push([left, vy], [right, vy]);
            });
            if (points.length >= 2) {
                const line = d3.line().x(d => x(d[0])).y(d => y(d[1])).curve(d3.curveStepAfter);
                const p = g.append('path')
                    .datum(points)
                    .attr('fill', 'none')
                    .attr('stroke', color)
                    .attr('stroke-width', 1.4)
                    .attr('opacity', 0.3)
                    .attr('d', line)
                    .attr('class', 'trace')
                    .attr('data-key', comp.key || null)
                    .attr('clip-path', `url(#${clipId})`);
                if (options.hoverable === true) {
                    p.style('cursor', 'crosshair');
                }
                // Optional vline marker
                if (Number.isFinite(comp.split_value)) {
                    const sx = x(comp.split_value);
                    if (sx >= 0 && sx <= innerW) {
                        g.append('line')
                            .attr('x1', sx)
                            .attr('x2', sx)
                            .attr('y1', 0)
                            .attr('y2', innerH)
                            .attr('stroke', color)
                            .attr('stroke-dasharray', '4,3')
                            .attr('stroke-width', 1.2)
                            .attr('opacity', 0.3)
                            .attr('clip-path', `url(#${clipId})`);
                    }
                    if (options.annotate === true) {
                        svg.append('text')
                            .attr('x', margin.left + Math.max(0, Math.min(innerW, x(comp.split_value))) + 4)
                            .attr('y', margin.top + 12)
                            .attr('fill', color)
                            .attr('font-size', 10)
                            .text(comp.split_label || '');
                    }
                }
            }
        });

        if (options.title) svg.append('text').attr('x', margin.left + 6).attr('y', 18).attr('fill', theme.text).attr('font-weight', 600).text(options.title);

        // Modal zoom-on-dblclick for step functions (optional)
        if (options && options.modalZoom === true) {
            // Attach to the container's dblclick
            d3.select(container).on('dblclick', () => {
                openModalZoom(container, (targetEl, currentDomain) => {
                    function renderWithDomain(domain) {
                        stepFunctions(targetEl, comps, bounds, { ...options, xDomain: domain, modalZoom: false });
                        const svgModal = d3.select(targetEl).select('svg');
                        const { innerW: mW, innerH: mH } = sized(targetEl);
                        const gModal = svgModal.select('g');
                        const xScale = d3.scaleLinear().domain((domain && domain.length === 2) ? domain : [bounds.min, bounds.max]).range([0, mW]);
                        const brush = d3.brushX()
                            .extent([[0, 0], [mW, mH]])
                            .on('end', (event) => {
                                if (!event.selection) return;
                                const [x0, x1] = event.selection;
                                const d0 = xScale.invert(x0);
                                const d1 = xScale.invert(x1);
                                // Re-render in place with new domain and reattach brush
                                renderWithDomain([d0, d1]);
                            });
                        gModal.append('g').attr('class', 'zoom-brush').call(brush);
                    }
                    renderWithDomain(currentDomain);
                });
            });
        }
    }

    api.lineChart = lineChart;
    api.multiLineChart = multiLineChart;
    api.barChart = barChart;
    api.stackedBars = stackedBars;
    api.scatter = scatter;
    api.scatterWithSelection = scatterWithSelection;
    api.stepFunctions = stepFunctions;
    // Expose shared helpers for pages to reuse
    api.getTooltip = getOrCreateTooltip;
    api.sizeOf = sized;
    return api;
})();

window.Charts = Charts;
