/* Shared helpers across pages */

window.UI = {
    async loadRunsTo(selectEl) {
        const runs = await API.getRuns();
        selectEl.innerHTML = '';
        runs.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r.run_id;
            opt.textContent = `Run ${r.run_id} (${r.n_events} events)`;
            selectEl.appendChild(opt);
        });
        return runs;
    },
    setOptions(selectEl, items) {
        selectEl.innerHTML = '';
        items.forEach(({ label, value }) => {
            const opt = document.createElement('option');
            opt.value = value;
            opt.textContent = label;
            selectEl.appendChild(opt);
        });
    },
    // Parsing utility shared across pages
    toFloat(v) {
        if (typeof v === 'number') return v;
        if (v == null) return NaN;
        if (Array.isArray(v) && v.length === 1) return UI.toFloat(v[0]);
        if (typeof v === 'string') {
            const s = v.trim().toLowerCase();
            if (s === 'inf' || s === '+inf' || s === 'infinity' || s === '+infinity') return Infinity;
            if (s === '-inf' || s === '-infinity') return -Infinity;
            const n = Number(v);
            return Number.isFinite(n) ? n : NaN;
        }
        const n = Number(v);
        return Number.isFinite(n) ? n : NaN;
    },
    /**
     * Compute [min,max] bounds for a list of intervals [[a,b,val], ...].
     * Applies a margin ratio to pad the domain.
     */
    computeBoundsFromIntervalsList(intervalsList, marginRatio = 0.05) {
        if (!Array.isArray(intervalsList) || intervalsList.length === 0) {
            return { min: -1, max: 1 };
        }
        const mins = [];
        const maxs = [];
        intervalsList.forEach(ib => {
            if (!Array.isArray(ib) || ib.length < 2) return;
            const a = UI.toFloat(ib[0]);
            const b = UI.toFloat(ib[1]);
            if (isFinite(a)) mins.push(a);
            if (isFinite(b)) maxs.push(b);
        });
        if (!(mins.length && maxs.length)) {
            return { min: -1, max: 1 };
        }
        let lo = Math.min(...mins);
        let hi = Math.max(...maxs);
        if (!(isFinite(lo) && isFinite(hi)) || lo === hi) {
            return { min: -1, max: 1 };
        }
        const span = hi - lo;
        const margin = span > 0 ? marginRatio * span : 1.0;
        return { min: lo - margin, max: hi + margin };
    },

    /**
     * Compute [min,max] bounds for step-function triples [[a,b,val], ...].
     * Handles open-ended (±∞) bounds which may arrive as `null` from JSON encoding.
     *
     * If any interval has a non-finite left bound, extends the left side by `extendRatio * span`.
     * Same for the right side.
     */
    computeBoundsFromIntervalsTriples(intervalsTriples, marginRatio = 0.05, extendRatio = 0.25) {
        if (!Array.isArray(intervalsTriples) || intervalsTriples.length === 0) {
            return { min: -1, max: 1 };
        }

        let minFinite = Infinity;
        let maxFinite = -Infinity;
        let hasOpenLeft = false;
        let hasOpenRight = false;

        for (const iv of intervalsTriples) {
            if (!Array.isArray(iv) || iv.length < 2) continue;
            const a = UI.toFloat(iv[0]);
            const b = UI.toFloat(iv[1]);
            if (Number.isFinite(a)) minFinite = Math.min(minFinite, a);
            else hasOpenLeft = true;
            if (Number.isFinite(b)) maxFinite = Math.max(maxFinite, b);
            else hasOpenRight = true;
        }

        if (!(Number.isFinite(minFinite) && Number.isFinite(maxFinite)) || minFinite === maxFinite) {
            return { min: -1, max: 1 };
        }

        const span = maxFinite - minFinite;
        const baseMargin = span > 0 ? marginRatio * span : 1.0;
        const extend = span > 0 ? Math.max(baseMargin, extendRatio * span) : 1.0;

        const min = minFinite - baseMargin - (hasOpenLeft ? extend : 0);
        const max = maxFinite + baseMargin + (hasOpenRight ? extend : 0);
        return { min, max };
    },

    /**
     * Compute bounds from multiple triples arrays (e.g. merged multi-tree view).
     */
    computeBoundsFromManyIntervalsTriples(intervalsTriplesList, marginRatio = 0.05, extendRatio = 0.25) {
        if (!Array.isArray(intervalsTriplesList) || intervalsTriplesList.length === 0) {
            return { min: -1, max: 1 };
        }
        // Flatten scan without allocating a big array.
        let minFinite = Infinity;
        let maxFinite = -Infinity;
        let hasOpenLeft = false;
        let hasOpenRight = false;

        for (const triples of intervalsTriplesList) {
            if (!Array.isArray(triples)) continue;
            for (const iv of triples) {
                if (!Array.isArray(iv) || iv.length < 2) continue;
                const a = UI.toFloat(iv[0]);
                const b = UI.toFloat(iv[1]);
                if (Number.isFinite(a)) minFinite = Math.min(minFinite, a);
                else hasOpenLeft = true;
                if (Number.isFinite(b)) maxFinite = Math.max(maxFinite, b);
                else hasOpenRight = true;
            }
        }

        if (!(Number.isFinite(minFinite) && Number.isFinite(maxFinite)) || minFinite === maxFinite) {
            return { min: -1, max: 1 };
        }

        const span = maxFinite - minFinite;
        const baseMargin = span > 0 ? marginRatio * span : 1.0;
        const extend = span > 0 ? Math.max(baseMargin, extendRatio * span) : 1.0;
        const min = minFinite - baseMargin - (hasOpenLeft ? extend : 0);
        const max = maxFinite + baseMargin + (hasOpenRight ? extend : 0);
        return { min, max };
    },

    /**
     * Calculate max iteration per tree from timeline events.
     * @param {Array} timeline - Timeline events from API
     * @param {number} epoch - Epoch to filter by
     * @param {Array<number>} treeIds - Tree IDs to calculate for
     * @returns {Map<number, number>} Map of treeId -> maxIter
     */
    calculateTreeMaxIters(timeline, epoch, treeIds) {
        const result = new Map();
        const wanted = Array.isArray(treeIds) ? treeIds.map(Number).filter(Number.isFinite) : [];
        const wantedSet = new Set(wanted);
        wanted.forEach(t => result.set(t, 0));

        if (!Array.isArray(timeline) || !Number.isFinite(+epoch) || wantedSet.size === 0) {
            return result;
        }

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
    },

    /**
     * Fetch identified lambda values for a tree, trying treeMaxIter and treeMaxIter+1.
     * The identified state (with final lambda values) is typically at treeMaxIter + 1
     * because timeline only has split events, but component_states may have
     * identification step logged at the next iteration.
     * @param {Function} fetchFn - Function to fetch data: (runId, epoch, treeId, iterNo) => Promise
     * @param {number} runId - Run ID
     * @param {number} epoch - Epoch
     * @param {number} treeId - Tree ID
     * @param {number} treeMaxIter - Max iteration from timeline
     * @returns {Promise<{lambda_plus: number|null, lambda_minus: number|null}>}
     */
    async fetchIdentifiedLambdas(fetchFn, runId, epoch, treeId, treeMaxIter) {
        try {
            // Try timeline-derived max iter first
            let identified = await fetchFn(runId, epoch, treeId, treeMaxIter);
            // Always try +1 to get the post-identification lambdas
            try {
                const retry = await fetchFn(runId, epoch, treeId, treeMaxIter + 1);
                if (retry && (retry.lambda_plus != null || retry.lambda_minus != null)) {
                    identified = retry;
                }
            } catch (e) {
                // If +1 doesn't exist, use the original (treeMaxIter)
            }
            return identified || { lambda_plus: null, lambda_minus: null };
        } catch (err) {
            console.warn(`Failed to fetch identified lambdas for tree ${treeId} at iter ${treeMaxIter}:`, err);
            return { lambda_plus: null, lambda_minus: null };
        }
    },

    /**
     * Calculate percentile from sorted array.
     * @param {Array<number>} sorted - Sorted array of numbers
     * @param {number} p - Percentile (0-100)
     * @returns {number}
     */
    percentile(sorted, p) {
        if (sorted.length === 0) return 0;
        if (sorted.length === 1) return sorted[0];
        const index = (p / 100) * (sorted.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        const weight = index - lower;
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    },

    /**
     * Create color scales from lambda values using percentile-based normalization.
     * Uses middle 95% (2.5th to 97.5th percentile) for the scale domain.
     * Outliers are colored with extreme colors.
     * @param {Array<number>} lambdaPlusValues - Lambda plus values
     * @param {Array<number>} lambdaMinusValues - Lambda minus values (optional)
     * @param {Object} options - Options
     * @param {Function} options.interpolatePlus - D3 interpolator for plus (default: d3.interpolateViridis)
     * @param {Function} options.interpolateMinus - D3 interpolator for minus (default: d3.interpolateInferno)
     * @returns {{colorScalePlus: Function|null, colorScaleMinus: Function|null}}
     */
    createLambdaColorScales(lambdaPlusValues, lambdaMinusValues = [], options = {}) {
        const { interpolatePlus = d3.interpolateViridis, interpolateMinus = d3.interpolateInferno } = options;

        const createScale = (values, interpolate) => {
            if (!values || values.length === 0) return null;
            const sorted = [...new Set(values)].sort((a, b) => a - b);

            if (sorted.length === 1) {
                // All values are the same, return constant color
                return () => interpolate(0.5);
            }

            // Use 2.5th and 97.5th percentiles for middle 95%
            const p2_5 = UI.percentile(sorted, 2.5);
            const p97_5 = UI.percentile(sorted, 97.5);

            // Create base scale for middle 95%
            const baseScale = d3.scaleSequential(interpolate).domain([p2_5, p97_5]);

            // Return function that handles outliers
            return (value) => {
                if (!isFinite(value)) return interpolate(0.5);

                // Outliers get extreme colors
                if (value < p2_5) {
                    return interpolate(0); // Minimum color for low outliers
                }
                if (value > p97_5) {
                    return interpolate(1); // Maximum color for high outliers
                }

                // Normal values use the percentile-based scale
                return baseScale(value);
            };
        };

        return {
            colorScalePlus: createScale(lambdaPlusValues, interpolatePlus),
            colorScaleMinus: createScale(lambdaMinusValues, interpolateMinus)
        };
    }
};

/**
 * Simple in-memory short-lived cache for time-series / evolution payloads.
 * Stores Promises to avoid duplicate inflight requests and supports manual invalidation.
 *
 * Usage:
 *   const key = UI.Cache.makeKey('backbone', runId, epoch, treeId, col);
 *   const data = await UI.Cache.fetchOrGet(key, () => API.getBackboneTiltEvolution(runId, epoch, treeId, col));
 */
UI.Cache = {
    _store: new Map(),
    _maxEntries: 600,
    makeKey(prefix, ...parts) {
        return `${prefix}::${parts.map(p => String(p)).join(':')}`;
    },
    has(key) {
        return this._store.has(key);
    },
    get(key) {
        const v = this._store.get(key);
        // If stored value is a Promise, return it (caller should await).
        return v;
    },
    set(key, value) {
        this._store.set(key, value);
        // Basic FIFO eviction to avoid unbounded growth during slider scrubbing
        while (this._store.size > this._maxEntries) {
            const firstKey = this._store.keys().next().value;
            if (firstKey == null) break;
            this._store.delete(firstKey);
        }
        return value;
    },
    delete(key) {
        return this._store.delete(key);
    },
    clear() {
        this._store.clear();
    },
    /**
     * Fetch using fetchFn if not present. If an inflight Promise exists, return it.
     * If fetchFn rejects, remove cached entry to allow retries.
     * @param {string} key
     * @param {Function} fetchFn - () => Promise
     */
    async fetchOrGet(key, fetchFn) {
        if (this._store.has(key)) {
            console.debug(`UI.Cache: cache hit for ${key}`);
            return await this._store.get(key);
        }
        const p = (async () => {
            try {
                const res = await fetchFn();
                return res;
            } catch (err) {
                // remove on failure so subsequent attempts can retry
                this._store.delete(key);
                throw err;
            }
        })();
        console.debug(`UI.Cache: fetching and caching ${key}`);
        this.set(key, p);
        return await p;
    }
};
