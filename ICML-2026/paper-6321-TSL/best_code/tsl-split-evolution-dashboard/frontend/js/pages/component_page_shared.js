(function () {
    function createChartContainer() {
        const div = document.createElement('div');
        div.className = 'chart';
        div.style.minHeight = '250px';
        div.style.width = '100%';
        div.style.maxWidth = '100%';
        div.style.overflow = 'hidden';
        div.style.boxSizing = 'border-box';
        div.style.border = '1px solid #e5e7eb';
        div.style.borderRadius = '4px';
        div.style.padding = '10px';
        return div;
    }

    function createLayoutManager(axesGrid) {
        const state = {
            key: null,
            mode: null,
            axes: []
        };

        return {
            reset() {
                state.key = null;
                state.mode = null;
                state.axes = [];
            },
            hasLayout() {
                return state.key !== null;
            },
            ensure({ layoutKey, modeKey = 'default', columnsMeta, setupGrid, createEntry }) {
                const desiredLength = columnsMeta.length;
                if (
                    state.key === layoutKey &&
                    state.mode === modeKey &&
                    state.axes.length === desiredLength
                ) {
                    return state.axes;
                }

                state.key = layoutKey;
                state.mode = modeKey;
                state.axes = [];

                axesGrid.innerHTML = '';
                axesGrid.style.width = '100%';
                axesGrid.style.boxSizing = 'border-box';
                if (setupGrid) {
                    setupGrid(axesGrid, modeKey, columnsMeta);
                } else {
                    axesGrid.style.display = 'grid';
                    axesGrid.style.gridTemplateColumns = '1fr';
                    axesGrid.style.gap = '20px';
                }

                columnsMeta.forEach((col, index) => {
                    const entry = createEntry(col, modeKey, {
                        axesGrid,
                        createChartContainer
                    }, index);
                    if (entry) {
                        state.axes.push(entry);
                    }
                });

                return state.axes;
            }
        };
    }

    function showStatus(axesGrid, layoutManager, message) {
        axesGrid.innerHTML = `<p>${message}</p>`;
        axesGrid.style.display = 'block';
        layoutManager.reset();
    }

    function getSelectedTrees(selectEl) {
        return Array.from(selectEl?.selectedOptions || [])
            .map(opt => parseInt(opt.value, 10))
            .filter(Number.isFinite);
    }

    function createRenderScheduler(renderFn) {
        let timer = null;
        return {
            schedule(delayMs = 100) {
                if (timer) clearTimeout(timer);
                timer = setTimeout(() => {
                    timer = null;
                    renderFn();
                }, delayMs);
            }
        };
    }

    function createRenderSequence() {
        let seq = 0;
        return {
            next() {
                return ++seq;
            },
            isCurrent(id) {
                return id === seq;
            }
        };
    }

    function createLambdaCache() {
        const cache = new Map();
        return {
            async ensure(runId, epoch, treeId, treeMaxIter, fetchFn, fallbackPlus = null, fallbackMinus = null) {
                const key = `${runId || 0}:${epoch || 0}:${treeId || 0}`;
                if (cache.has(key)) {
                    return cache.get(key);
                }

                let lambdaPlus = fallbackPlus;
                let lambdaMinus = fallbackMinus;
                if (lambdaPlus == null && lambdaMinus == null && typeof fetchFn === 'function') {
                    const identified = await UI.fetchIdentifiedLambdas(fetchFn, runId, epoch, treeId, treeMaxIter);
                    lambdaPlus = identified?.lambda_plus ?? null;
                    lambdaMinus = identified?.lambda_minus ?? null;
                }

                const payload = { lambda_plus: lambdaPlus, lambda_minus: lambdaMinus };
                cache.set(key, payload);
                return payload;
            }
        };
    }

    function createAsyncCache() {
        const store = new Map();
        return {
            async getOrCreate(key, fetcher) {
                if (!store.has(key)) {
                    const promise = Promise.resolve()
                        .then(() => fetcher())
                        .catch(err => {
                            store.delete(key);
                            throw err;
                        });
                    store.set(key, promise);
                }
                return store.get(key);
            },
            delete(key) {
                store.delete(key);
            },
            clear() {
                store.clear();
            }
        };
    }

    window.ComponentPageShared = {
        createChartContainer,
        createLayoutManager,
        showStatus,
        getSelectedTrees,
        createRenderScheduler,
        createRenderSequence,
        createLambdaCache,
        createAsyncCache
    };
})();
