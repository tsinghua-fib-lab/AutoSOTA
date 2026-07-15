/* Minimal API client for the FastAPI backend */

const API = {
    async getRuns() {
        const res = await fetch('/api/runs');
        if (!res.ok) throw new Error('Failed to fetch runs');
        return res.json();
    },

    async getTimeline(runId) {
        const res = await fetch(`/api/run/${runId}/timeline`);
        if (!res.ok) throw new Error('Failed to fetch timeline');
        return res.json();
    },

    async getColumns(runId) {
        const res = await fetch(`/api/run/${runId}/columns`);
        if (!res.ok) throw new Error('Failed to fetch columns');
        return res.json();
    },

    async getColumnsSummary(runId) {
        const res = await fetch(`/api/run/${runId}/columns_summary`);
        if (!res.ok) throw new Error('Failed to fetch columns summary');
        return res.json();
    },

    async getLearning(runId) {
        const res = await fetch(`/api/run/${runId}/learning`);
        if (!res.ok) throw new Error('Failed to fetch learning');
        return res.json();
    },

    async getConvergence(runId) {
        const res = await fetch(`/api/run/${runId}/convergence`);
        if (!res.ok) throw new Error('Failed to fetch convergence');
        return res.json();
    },

    async getGridErrors(runId) {
        const res = await fetch(`/api/run/${runId}/grid_errors`);
        if (!res.ok) throw new Error('Failed to fetch grid errors');
        return res.json();
    },

    async getEpochsTrees(runId) {
        const res = await fetch(`/api/run/${runId}/epochs_trees`);
        if (!res.ok) throw new Error('Failed to fetch epochs/trees');
        return res.json();
    },

    async getFComponentEpochsTrees(runId) {
        const res = await fetch(`/api/run/${runId}/f_component_epochs_trees`);
        if (!res.ok) {
            const errorText = await res.text().catch(() => '');
            throw new Error(`Failed to fetch f_component epochs/trees: ${res.status} ${res.statusText}${errorText ? ' - ' + errorText : ''}`);
        }
        return res.json();
    },

    async getTreeEvolution(runId, epoch, treeId, iteration) {
        const params = new URLSearchParams({ epoch, tree_id: treeId, iteration });
        const res = await fetch(`/api/run/${runId}/tree_evolution?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch tree evolution');
        return res.json();
    },



    async getIdentified(runId, epoch = 0, treeId = 0) {
        const params = new URLSearchParams({ epoch, tree_id: treeId });
        const res = await fetch(`/api/run/${runId}/identified_components?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch identified components');
        return res.json();
    },

    async getIdentifiedAll(runId, epoch, maxTrees = 50) {
        const params = new URLSearchParams({ epoch, max_trees: maxTrees });
        const res = await fetch(`/api/run/${runId}/identified_components_all?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch identified components for all trees');
        return res.json();
    },

    async getCombinedEpochs() {
        const res = await fetch('/api/epochs');
        if (!res.ok) throw new Error('Failed to fetch epochs');
        return res.json();
    },

    async getCombinedForEpoch(epoch) {
        const res = await fetch(`/api/combined/epoch/${epoch}`);
        if (!res.ok) throw new Error('Failed to fetch combined epoch');
        return res.json();
    },

    async getUnifiedTreeComponents(runId, epoch, iteration = 0, identified = false, selectedTrees = null) {
        const params = new URLSearchParams({
            epoch,
            iteration,
            identified: identified ? 'true' : 'false'
        });
        if (selectedTrees) {
            params.append('selected_trees', selectedTrees);
        }
        const res = await fetch(`/api/run/${runId}/unified_tree_components?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch unified tree components');
        return res.json();
    },

    async getEpochScalings(runId) {
        const res = await fetch(`/api/run/${runId}/scalings`);
        if (!res.ok) throw new Error('Failed to fetch epoch scalings');
        return res.json();
    },

    async getEpochEnergy(runId) {
        const res = await fetch(`/api/run/${runId}/energy`);
        if (!res.ok) throw new Error('Failed to fetch epoch energy');
        return res.json();
    },

    async getBackboneTiltEvolution(runId, epoch, treeId, col, startIter = 0, endIter = null) {
        const params = new URLSearchParams({ epoch, tree_id: treeId, col, start_iter: startIter });
        if (endIter !== null) params.append('end_iter', endIter);
        const res = await fetch(`/api/run/${runId}/backbone_tilt_evolution?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch backbone/tilt evolution');
        return res.json();
    },

    async getBackboneTiltEvolutionAllColumns(runId, epoch, treeId, startIter = 0, endIter = null) {
        const params = new URLSearchParams({ epoch, tree_id: treeId, start_iter: startIter });
        if (endIter !== null) params.append('end_iter', endIter);
        const res = await fetch(`/api/run/${runId}/backbone_tilt_evolution_all_columns?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch backbone/tilt evolution (all columns)');
        return res.json();
    },

    async getFComponentEvolution(runId, epoch, treeId, startIter = 0, endIter = null) {
        const params = new URLSearchParams({ epoch, tree_id: treeId, start_iter: startIter });
        if (endIter !== null) params.append('end_iter', endIter);
        const res = await fetch(`/api/run/${runId}/f_component_evolution?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch f+/f- component evolution');
        return res.json();
    },

    async getFComponentPerAxis(runId, epoch, treeId, iterNo) {
        const res = await fetch(`/api/run/${runId}/f_component_per_axis?epoch=${epoch}&tree_id=${treeId}&iter_no=${iterNo}`);
        if (!res.ok) throw new Error('Failed to fetch f+/f- per axis');
        return res.json();
    },

    async getTensorLambdas(runId, epoch, maxTrees = 500) {
        const params = new URLSearchParams({ epoch, max_trees: maxTrees });
        const res = await fetch(`/api/run/${runId}/tensor_lambdas?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch product lambdas');
        return res.json();
    },

    async getCombinationChoice(runId, epoch) {
        const params = new URLSearchParams({ epoch });
        const res = await fetch(`/api/run/${runId}/combination_choice?${params.toString()}`);
        if (!res.ok) {
            // Return null if not found / no data
            return null;
        }
        return res.json();
    },

    async getFComponentPerAxisMulti(runId, epoch, iterNo, treeIds) {
        const treeIdsCsv = Array.isArray(treeIds) ? treeIds.join(',') : String(treeIds || '');
        const params = new URLSearchParams({ epoch, iter_no: iterNo, tree_ids: treeIdsCsv });
        const res = await fetch(`/api/run/${runId}/f_component_per_axis_multi?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch f+/f- per axis (multi-tree)');
        return res.json();
    },

    async getComponentDecomposition(runId, epoch, treeId, iterNo) {
        const params = new URLSearchParams({ epoch, tree_id: treeId, iter_no: iterNo });
        const res = await fetch(`/api/run/${runId}/component_decomposition?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch component decomposition');
        return res.json();
    },

    async getComponentDecompositionMulti(runId, epoch, iterNo, treeIds) {
        const treeIdsCsv = Array.isArray(treeIds) ? treeIds.join(',') : String(treeIds || '');
        const params = new URLSearchParams({ epoch, iter_no: iterNo, tree_ids: treeIdsCsv });
        const res = await fetch(`/api/run/${runId}/component_decomposition_multi?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch component decomposition (multi-tree)');
        return res.json();
    }
};

window.API = API;
