classdef MES_RET < Algorithm
% <Multi-task/Many-task> <Single-objective> <None/Constrained>

%------------------------------- Reference --------------------------------
% @InProceedings{Li2026MES-RET,
%   author    = {Li, Yanchi and Liu, Jiao and Gong, Wenyin and Gu, Qiong and Zhao, Yue and Ong, Yew-Soon},
%   booktitle = {Forty-third International Conference on Machine Learning},
%   title     = {Breaking Multi-Task Curse: Reward-Weighted Evolution for Black-Box Many-Task Optimization},
%   year      = {2026},
%   url       = {https://openreview.net/forum?id=lkGnJhXUNu},
% }
%--------------------------------------------------------------------------

%------------------------------- Copyright --------------------------------
% Copyright (c) Yanchi Li. You are free to use the MToP for research
% purposes. All publications which use this platform should acknowledge
% the use of MToP and cite as "Y. Li, W. Gong, T. Zhang, F. Ming,
% S. Li, Q. Gu, and Y.-S. Ong, MToP: A MATLAB Benchmarking Platform for
% Evolutionary Multitasking, 2023, arXiv:2312.08134"
%--------------------------------------------------------------------------

properties (SetAccess = public)
    sigma0 = 0.3
    tau = 1
end

methods
    function Parameter = getParameter(Algo)
        Parameter = {'sigma0', num2str(Algo.sigma0), ...
                'tau', num2str(Algo.tau)};
    end

    function Algo = setParameter(Algo, Parameter)
        Algo.sigma0 = str2double(Parameter{1});
        Algo.tau = str2double(Parameter{2});
    end

    function run(Algo, Prob)
        % Initialization
        CMA = Algo.InitCMA(Prob, Prob.N);
        Reward = ones(1, Prob.T) ./ Prob.T;

        % Main Loop
        while Algo.notTerminated(Prob, CMA.sample)
            % Self Evolution
            for t = 1:Prob.T
                if CMA.StopFlag(t), continue; end
                CMA.PreviousPara{t} = Algo.SavePara(CMA, t);
                CMA = Algo.SamplingAndTransfer(Prob, CMA, t);
                CMA = Algo.ParameterUpdate(Prob, CMA, t);
            end

            % Reward Calculation
            if rand() < 1 - Algo.FE / Prob.maxFE
                Reward = Algo.ComputeRewardFit(Prob, CMA);
            else
                Reward = Algo.ComputeRewardDiv(Prob, CMA);
            end

            % Reward-weighted Knowledge Aggregation on Mean and Variance
            for j = 1:max(Prob.D)
                idx = ~CMA.StopFlag & ~isnan(CMA.std_ratio(:, j))' & ~isnan(CMA.mdec_matrix(:, j))';
                if isempty(find(idx, 1))
                    CMA.agg_std_ratio(j) = NaN;
                    CMA.agg_mdec(j) = NaN;
                else
                    CMA.agg_std_ratio(j) = sum(Reward(idx) ./ sum(Reward(idx)) .* CMA.std_ratio(idx, j)');
                    CMA.agg_mdec(j) = sum(Reward(idx) ./ sum(Reward(idx)) .* CMA.mdec_matrix(idx, j)');
                end
            end
            idx = ~CMA.StopFlag;
            CMA.agg_sigma_ratio = sum(Reward(idx) ./ sum(Reward(idx)) .* CMA.sigma_ratio(idx)');

            % Reward-weighted Evaluation
            for x = 1:Prob.T
                t = RouletteSelection(Reward);
                if CMA.StopFlag(t), continue; end
                CMA.PreviousPara{t} = Algo.SavePara(CMA, t);
                CMA = Algo.SamplingAndTransfer(Prob, CMA, t);
                CMA = Algo.ParameterUpdate(Prob, CMA, t);
            end

            % Check Stopping Criteria
            CMA = Algo.CheckStop(Prob, CMA);
        end
    end

    function RewardFit = ComputeRewardFit(Algo, Prob, CMA)
        % RewardFit: Reward for objective improvement
        % Strategy: Min-Max Normalization -> Softmax
        improveFit = zeros(1, Prob.T);
        for t = 1:Prob.T
            if CMA.StopFlag(t)
                improveFit(t) = 0;
                continue;
            end

            if CMA.gBest{t}.CV <= 0
                old_best = CMA.PreviousPara{t}.fitness(2);
                init_best = CMA.obj_init{t};
                new_best = CMA.gBest{t}.Obj;
            else
                old_best = CMA.PreviousPara{t}.fitness(1);
                init_best = CMA.cv_init{t};
                new_best = CMA.gBest{t}.CV;
            end

            % Calculate improvement
            improveFit(t) = max(0, old_best - new_best) / (abs(init_best - old_best) +1e-12);
        end

        % Normalization
        idx = ~CMA.StopFlag;
        if sum(idx) == 0
            RewardFit = ones(1, Prob.T) / Prob.T;
            return;
        end

        vals = improveFit(idx);
        min_v = min(vals);
        max_v = max(vals);
        range = max_v - min_v;

        noise_threshold = 1e-9;
        if range < noise_threshold
            norm_vals = zeros(size(vals));
        else
            norm_vals = (vals - min_v) / range;
        end

        % Softmax transformation
        exps = exp(norm_vals);
        probs = exps / sum(exps);

        RewardFit = zeros(1, Prob.T);
        RewardFit(idx) = probs;

        if any(isnan(RewardFit))
            RewardFit = ones(1, Prob.T) / Prob.T;
        end
    end

    function RewardDiv = ComputeRewardDiv(Algo, Prob, CMA)
        % RewardDiv: Reward for diversity
        % Normalized trace of covariance matrix
        for t = 1:Prob.T
            if CMA.StopFlag(t)
                diversity(t) = 0;
                continue;
            end
            sigma = CMA.PreviousPara{t}.sigma;
            traceC = sum(CMA.PreviousPara{t}.std.^2);
            diversity(t) = sigma * traceC / CMA.n{t};
            sigma = CMA.sigma{t};
            traceC = trace(CMA.C{t});
            diversity(t) = diversity(t) + sigma * traceC / CMA.n{t};
        end
        diversity = (diversity - min(diversity)) / (max(diversity) - min(diversity) +1e-12);
        sumDiv = sum(diversity);
        if sumDiv < 1e-12
            RewardDiv = ones(1, Prob.T) / Prob.T;
        else
            RewardDiv = diversity / sumDiv;
        end
        if any(isnan(RewardDiv))
            RewardDiv = ones(1, Prob.T) / Prob.T;
        end
    end

    function CMA = SamplingAndTransfer(Algo, Prob, CMA, t)
        % Sample new solutions (Standard Sampling for all dims)
        for i = 1:CMA.lambda{t}
            CMA.sample{t}(i).Dec = CMA.mDec{t} + CMA.sigma{t} * (CMA.B{t} * (CMA.D{t} .* randn(CMA.n{t}, 1)))';
        end

        % Knowledge transfer
        if Algo.tau > 0 && Algo.Gen > 10
            if Algo.tau >= 1, tr_num = round(Algo.tau);
            elseif rand() < Algo.tau, tr_num = 1;
            else tr_num = 0;
            end

            eff_dim = CMA.n{t};
            % Mean: Mean decision variable sampling
            m = CMA.agg_mdec(1:eff_dim);
            if any(isnan(m)), m = CMA.mDec{t}; end
            % Calculate the average step size in the effective dimension
            decs = CMA.sample{t}.Decs;
            mStep_partial = mean(sqrt(sum((decs(:, 1:eff_dim) - CMA.mDec{t}(1:eff_dim)).^2, 2)));
            % Generate tr_num samples toward the mean in the effective dimension
            for i = 1:tr_num
                temp = (CMA.B{t} * (CMA.D{t} .* randn(CMA.n{t}, 1)))';
                u = m(1:eff_dim) - CMA.mDec{t}(1:eff_dim) + CMA.sigma{t} * temp(1:eff_dim);
                CMA.sample{t}(i).Dec(1:eff_dim) = CMA.mDec{t}(1:eff_dim) + u / norm(u) * mStep_partial;
            end

            % Covariance: Distribution variation sampling
            v = CMA.agg_sigma_ratio .* CMA.agg_std_ratio(1:CMA.n{t});
            if any(isnan(v)), v = ones(1, CMA.n{t}); end
            if CMA.n{t} > eff_dim
                v(eff_dim + 1:end) = 1;
            end
            % Generate samples using the transferred variance information
            for i = tr_num + 1:tr_num * 2
                u = v .* CMA.sigma{t} .* sqrt(diag(CMA.C{t}))' .* randn(1, CMA.n{t});
                CMA.sample{t}(i).Dec = CMA.mDec{t} + u;
            end
        end

        [CMA.sample{t}, CMA.rank{t}] = Algo.EvaluationAndSort(CMA.sample{t}, Prob, t, CMA.cv_max{t});
        % Update the global best solution
        if ~isempty(CMA.gBest{t}) && (Algo.Best{t}.CV < CMA.gBest{t}.CV || ...
                (Algo.Best{t}.CV == CMA.gBest{t}.CV && Algo.Best{t}.Obj < CMA.gBest{t}.Obj))
            CMA.gBest{t} = Algo.Best{t};
        end
    end

    function CMA = InitCMA(Algo, Prob, lambda)
        % Initialize CMA-ES parameters for T tasks
        for t = 1:Prob.T
            CMA.n{t} = Prob.D(t); % dimension
            CMA.lambda{t} = lambda; % sample points number
            CMA.mu{t} = round(CMA.lambda{t} / 2); % effective solutions number
            CMA.weights{t} = log(CMA.mu{t} + 0.5) - log(1:CMA.mu{t});
            CMA.weights{t} = CMA.weights{t} ./ sum(CMA.weights{t}); % weights
            CMA.mueff = 1 / sum(CMA.weights{t}.^2); % variance effective selection mass
            % expectation of the Euclidean norm of a N(0,I) distributed random vector
            CMA.chiN{t} = sqrt(CMA.n{t}) * (1 - 1 / (4 * CMA.n{t}) + 1 / (21 * CMA.n{t}^2));
            CMA.hth{t} = (1.4 + 2 / (CMA.n{t} + 1)) * CMA.chiN{t};
            % Step size control parameters
            CMA.cs{t} = (CMA.mueff + 2) / (CMA.n{t} + CMA.mueff + 5);
            CMA.damps{t} = 1 + CMA.cs{t} + 2 * max(sqrt((CMA.mueff - 1) / (CMA.n{t} + 1)) - 1, 0);
            % Covariance update parameters
            CMA.cc{t} = (4 + CMA.mueff / CMA.n{t}) / (4 + CMA.n{t} + 2 * CMA.mueff / CMA.n{t});
            CMA.c1{t} = 2 / ((CMA.n{t} + 1.3)^2 + CMA.mueff);
            CMA.cmu{t} = min(1 - CMA.c1{t}, 2 * (CMA.mueff - 2 + 1 / CMA.mueff) / ((CMA.n{t} + 2)^2 + 2 * CMA.mueff / 2));
            % Initialization
            CMA.mDec{t} = initESMean(Prob, t);
            Algo.Mean{t} = CMA.mDec{t};
            CMA.ps{t} = zeros(CMA.n{t}, 1);
            CMA.pc{t} = zeros(CMA.n{t}, 1);
            CMA.B{t} = eye(CMA.n{t}, CMA.n{t});
            CMA.D{t} = ones(CMA.n{t}, 1);
            CMA.C{t} = CMA.B{t} * diag(CMA.D{t}.^2) * CMA.B{t}';
            CMA.invsqrtC{t} = CMA.B{t} * diag(CMA.D{t}.^-1) * CMA.B{t}';
            CMA.sigma{t} = Algo.sigma0 * initESSigmaScale(Prob, t);
            CMA.eigenFE{t} = 0;
            for i = 1:CMA.lambda{t}
                CMA.sample{t}(i) = Individual();
            end
            CMA.PreviousPara{t} = [];
            CMA.obj_init{t} = 0;
            CMA.cv_init{t} = 0;
            CMA.cv_max{t} = 0;
        end

        % Initialize knowledge extraction parameters
        CMA.sigma_ratio = ones(Prob.T, 1);
        CMA.std_ratio = nan(Prob.T, max(Prob.D));
        CMA.mdec_matrix = nan(Prob.T, max(Prob.D));
        CMA.StopFlag = false(1, Prob.T);

        % Initial sampling
        for t = 1:Prob.T
            CMA.gBest{t} = [];
            CMA = Algo.SamplingAndTransfer(Prob, CMA, t);
            CMA = Algo.ParameterUpdate(Prob, CMA, t);
            temp_cv = CMA.sample{t}.CVs;
            [~, idx] = sort(temp_cv);
            CMA.cv_max{t} = temp_cv(idx(round(0.2 * length(temp_cv))));
            CMA.gBest{t} = Algo.Best{t};
            CMA.obj_init{t} = CMA.gBest{t}.Obj;
            CMA.cv_init{t} = CMA.gBest{t}.CV;
        end
    end

    function CMA = ParameterUpdate(Algo, Prob, CMA, t)
        % Update CMA parameters
        % Update mean decision variables
        oldDec = CMA.mDec{t};
        rankDecs = CMA.sample{t}(CMA.rank{t}(1:CMA.mu{t})).Decs;
        rankDecs = rankDecs(:, 1:CMA.n{t});
        CMA.mDec{t} = CMA.weights{t} * rankDecs;
        Algo.Mean{t} = CMA.mDec{t};
        % Update evolution paths
        CMA.ps{t} = (1 - CMA.cs{t}) * CMA.ps{t} + sqrt(CMA.cs{t} * (2 - CMA.cs{t}) * CMA.mueff) * CMA.invsqrtC{t} * (CMA.mDec{t} - oldDec)' / CMA.sigma{t};
        hsig = norm(CMA.ps{t}) / sqrt(1 - (1 - CMA.cs{t})^(2 * (ceil((Algo.FE - CMA.lambda{t} * (t - 1)) / (CMA.lambda{t} * Prob.T)) + 1))) < CMA.hth{t};
        CMA.pc{t} = (1 - CMA.cc{t}) * CMA.pc{t} + hsig * sqrt(CMA.cc{t} * (2 - CMA.cc{t}) * CMA.mueff) * (CMA.mDec{t} - oldDec)' / CMA.sigma{t};
        % Update covariance matrix
        artmp = (rankDecs - repmat(oldDec, CMA.mu{t}, 1))' / CMA.sigma{t};
        delta = (1 - hsig) * CMA.cc{t} * (2 - CMA.cc{t});
        CMA.C{t} = (1 - CMA.c1{t} - CMA.cmu{t}) * CMA.C{t} + CMA.c1{t} * (CMA.pc{t} * CMA.pc{t}' + delta * CMA.C{t}) + CMA.cmu{t} * artmp * diag(CMA.weights{t}) * artmp';
        % Update step size
        CMA.sigma{t} = CMA.sigma{t} * exp(CMA.cs{t} / CMA.damps{t} * (norm(CMA.ps{t}) / CMA.chiN{t} - 1));
        % Check distribution correctness
        if (Algo.FE - CMA.lambda{t} * (t - 1)) - CMA.eigenFE{t} > (CMA.lambda{t} * Prob.T) / (CMA.c1{t} + CMA.cmu{t}) / CMA.n{t} / 10 % to achieve O(N^2)
            CMA.eigenFE{t} = Algo.FE;
            if ~(all(~isnan(CMA.C{t}), 'all') && all(~isinf(CMA.C{t}), 'all'))
                CMA.StopFlag(t) = true;
            else
                CMA.C{t} = triu(CMA.C{t}) + triu(CMA.C{t}, 1)'; % enforce symmetry
                [CMA.B{t}, CMA.D{t}] = eig(CMA.C{t}); % eigen decomposition, B==normalized eigenvectors
                if min(diag(CMA.D{t})) < 0
                    CMA.StopFlag(t) = true;
                else
                    CMA.D{t} = sqrt(diag(CMA.D{t})); % D contains standard deviations now
                end
            end
            if CMA.StopFlag(t)
                return;
            end
            CMA.invsqrtC{t} = CMA.B{t} * diag(CMA.D{t}.^-1) * CMA.B{t}';
        end

        % Update knowledge extraction parameters
        if ~isempty(CMA.PreviousPara{t})
            CMA.sigma_ratio(t) = CMA.sigma{t} / CMA.PreviousPara{t}.sigma;
            CMA.std_ratio(t, 1:CMA.n{t}) = sqrt(diag(CMA.C{t}))' ./ CMA.PreviousPara{t}.std;
            CMA.mdec_matrix(t, 1:CMA.n{t}) = CMA.mDec{t};
        end
    end

    function Para = SavePara(Algo, CMA, t)
        % Save parameters for the next generation
        Para = struct();
        Para.std = sqrt(diag(CMA.C{t}))';
        Para.sigma = CMA.sigma{t};
        bestObj = CMA.gBest{t}.Obj;
        bestCV = CMA.gBest{t}.CV;
        Para.fitness = [bestCV; bestObj];
    end

    function CMA = CheckStop(Algo, Prob, CMA)
        % Check stopping criteria (sigma * max(pc, std) < 1e-12)
        for t = 1:Prob.T
            if all(CMA.sigma{t} * (max(abs(CMA.pc{t}), sqrt(diag(CMA.C{t})))) < 1e-12)
                CMA.StopFlag(t) = true;
            end
        end
        % If all tasks stop, reset the stop flags to continue
        if all(CMA.StopFlag)
            for t = 1:Prob.T
                CMA.StopFlag(t) = false;
            end
        end
    end

    function [sample, rank] = EvaluationAndSort(Algo, sample, Prob, t, cv_max)
        % Evaluation
        sample = Algo.Evaluation(sample, Prob, t);

        % Epsilon constraint handling for constrained problems
        CVs = sample.CVs;
        if Algo.FE < 0.3 * Prob.maxFE && cv_max > 0
            Ep = cv_max * ((1 - Algo.FE / (0.3 * Prob.maxFE))^8);
            CVs(CVs < Ep) = 0;
        end

        if Prob.Bounded
            if any(CVs > 0)
                % Penalty on CV
                currentCVs = CVs;
                cvScale = max(currentCVs);
                if cvScale < 1, cvScale = 1; end
                penalty = zeros(length(sample), 1);
                for i = 1:length(sample)
                    tempDec = max(0, min(1, sample(i).Dec));
                    violation = sum((sample(i).Dec - tempDec).^2);
                    penalty(i) = violation * cvScale;
                end
                % get rank based on constraint violation
                [~, rank] = sortrows([CVs + penalty, sample.Objs], [1, 2]);
            else
                % Penalty method
                currentObjs = sample.Objs;
                objScale = max(abs(currentObjs));
                if objScale < 1, objScale = 1; end
                penalty = zeros(length(sample), 1);
                for i = 1:length(sample)
                    tempDec = max(0, min(1, sample(i).Dec));
                    violation = sum((sample(i).Dec - tempDec).^2);
                    penalty(i) = violation * objScale;
                end
                % get rank based on constraint and objective
                [~, rank] = sortrows([CVs, sample.Objs + penalty], [1, 2]);
            end
        else
            [~, rank] = sortrows([CVs, sample.Objs], [1, 2]);
        end
    end
end
end
