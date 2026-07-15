(function () {
    function sanitizeMaxTrees(value, fallback = 500) {
        const num = Number(value);
        if (!Number.isFinite(num)) return fallback;
        // No upper cap: accept any finite positive integer
        return Math.max(1, Math.trunc(num));
    }

    function makeScatterSeries(rows, choices) {
        const palette = (Charts.__theme && Charts.__theme.palette) || ['#3b82f6', '#55A868', '#DD8452'];
        const validRows = rows.filter(r => Number.isFinite(+r.lambda_plus) && Number.isFinite(+r.lambda_minus));
        const allData = validRows.map(r => ({
            tree_id: r.tree_id,
            lambda_plus: +r.lambda_plus,
            lambda_minus: +r.lambda_minus,
            tooltipData: {
                'Tree': r.tree_id,
                'λ+': r.lambda_plus,
                'λ-': r.lambda_minus,
                'Iter': r.iter_no
            }
        }));

        const candidatesSet = new Set((choices && choices.candidates) ? choices.candidates.map(c => c.tree_id) : []);
        const bestId = choices && Number.isFinite(+choices.best_index) ? +choices.best_index : null;

        const candidateData = allData.filter(d => candidatesSet.has(d.tree_id)).map(d => ({ ...d, score: (choices && choices.candidates && choices.candidates.find(c => c.tree_id === d.tree_id))?.score }));
        const bestData = allData.filter(d => bestId !== null && d.tree_id === bestId);
        const restData = allData.filter(d => !candidatesSet.has(d.tree_id) && !(bestId !== null && d.tree_id === bestId));

        const series = [
            { name: 'lambdas', color: palette[0], data: restData, r: 2.4 },
        ];
        if (candidateData.length) {
            series.push({ name: 'candidates', color: palette[1], data: candidateData, r: 3.2 });
        }
        if (bestData.length) {
            // Use yellow for the best index marker
            series.push({ name: 'best', color: '#f59e0b', data: bestData, r: 4.0 });
        }
        return series;
    }

    function renderScatter(container, rows, logScale = false, choices = null) {
        const series = makeScatterSeries(rows, choices);
        if (!series.length || series.every(s => !s.data.length)) {
            container.innerHTML = '<p>No lambda data for this stage.</p>';
            return;
        }
        // Limit the rendered scatter size to avoid overly large plots on wide screens.
        Charts.scatter(container, series, {
            xKey: 'lambda_plus',
            yKey: 'lambda_minus',
            title: 'λ⁺ vs λ⁻ per product',
            log: logScale,
            square: true,
            // cap width to keep the chart compact
            maxWidth: 820
        });
    }

    function computeRangeLabel(values) {
        if (!values.length) return 'n/a';
        const min = Math.min(...values);
        const max = Math.max(...values);
        if (!Number.isFinite(min) || !Number.isFinite(max)) return 'n/a';
        return `[${min.toFixed(4)}, ${max.toFixed(4)}]`;
    }

    function renderHistogramSummary(container, rows) {
        if (!rows.length) {
            container.innerHTML = '<p>No products available.</p>';
            return;
        }

        const validPlus = [];
        const validMinus = [];
        let bothFinite = 0;
        rows.forEach(r => {
            const lp = Number(r.lambda_plus);
            const lm = Number(r.lambda_minus);
            if (Number.isFinite(lp)) validPlus.push(lp);
            if (Number.isFinite(lm)) validMinus.push(lm);
            if (Number.isFinite(lp) && Number.isFinite(lm)) bothFinite++;
        });

        const summary = document.createElement('div');
        summary.style.fontSize = '13px';
        summary.style.color = '#475569';
        summary.innerHTML = [
            '<b>Summary</b>',
            `λ⁺ range ${computeRangeLabel(validPlus)}`,
            `λ⁻ range ${computeRangeLabel(validMinus)}`,
            `products plotted ${bothFinite}/${rows.length}`
        ].join(' · ');

        container.innerHTML = '';
        container.appendChild(summary);
    }

    window.LambdaScatterHelpers = {
        sanitizeMaxTrees,
        renderScatter,
        renderHistogramSummary
    };
})();
