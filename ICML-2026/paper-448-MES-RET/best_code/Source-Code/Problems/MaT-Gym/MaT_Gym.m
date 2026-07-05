classdef MaT_Gym < Problem
% <Many-task> <Single-objective> <None>

% MaT_Gym: Many-Task Reinforcement Learning Benchmark
%
% Key Features:
%   - Controlled Randomness: Random seed is generated in MATLAB and passed to Python.
%   - Dynamic Neural Network Architecture.
%   - Online Input Normalization (Welford's Algorithm).

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

properties (Access = public)
    % --- Hyperparameters ---
    hiddenLayers = 2; % Network depth
    hiddenSize = 16; % Network width
    jointSpaceDim = 16 * (16 + 1) * (2 - 1) + 16; % Calculated based on hidden size and layers
    numRollouts = 3; % Evaluations per individual

    tasks % Metadata storage

    % --- Welford Statistics (Online Normalization) ---
    ObsMean = {};
    ObsStd = {};
    ObsM2 = {};
    ObsCount = {};

    % --- Initialization ---
    initSolutionMean = {};
    initSigmaScale = {};
end

methods
    function Prob = MaT_Gym(varargin)
        Prob = Prob@Problem(varargin);
        Prob.maxFE = 18 * 50 * 500;
        Prob.ReEvalBest = true;
        Prob.Bounded = false;
    end

    function Parameter = getParameter(Prob)
        Parameter = {'HiddenLayers', num2str(Prob.hiddenLayers), ...
                'HiddenSize', num2str(Prob.hiddenSize), ...
                'NumRollouts', num2str(Prob.numRollouts)};
        Parameter = [Prob.getRunParameter(), Parameter];
    end

    function Prob = setParameter(Prob, Parameter)
        Prob.hiddenLayers = str2double(Parameter{3});
        Prob.hiddenSize = str2double(Parameter{4});
        % Calculate joint space dimension based on hidden size and layers
        Prob.jointSpaceDim = Prob.hiddenSize * (Prob.hiddenSize + 1) * (Prob.hiddenLayers - 1) + Prob.hiddenSize;
        Prob.numRollouts = str2double(Parameter{5});
        Prob.setRunParameter(Parameter(1:2));
    end

    function setTasks(Prob)
        if count(py.sys.path, 'Problems/Test/MaT-Gym') == 0
            py.sys.path().append('Problems/Test/MaT-Gym');
        end
        try
            py.importlib.reload(py.importlib.import_module('gym_runner'));
        catch
            py.importlib.import_module('gym_runner');
        end

        taskNames = {"MountainCarContinuous-v0", "MountainCar-v0", ...
                "Pendulum-v1", "CartPole-v1", "Acrobot-v1", ...
                "LunarLander-v3", "BipedalWalker-v3", ...
                "InvertedPendulum-v5", "InvertedDoublePendulum-v5", ...
                "Reacher-v5", "Pusher-v5", ...
                "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", ...
                "Swimmer-v5", "Ant-v5", ...
                "Humanoid-v5", "HumanoidStandup-v5"};

        Prob.T = length(taskNames);
        Prob.initSolutionMean = cell(1, Prob.T);

        Prob.ObsMean = cell(1, Prob.T);
        Prob.ObsStd = cell(1, Prob.T);
        Prob.ObsM2 = cell(1, Prob.T);
        Prob.ObsCount = cell(1, Prob.T);

        gymModule = py.importlib.import_module('gymnasium');
        Prob.tasks = repmat(struct('name', '', 'obsDim', 0, 'actDim', 0, ...
            'paramCount', 0), 1, Prob.T);

        for i = 1:Prob.T
            envName = taskNames{i};
            Prob.tasks(i).name = envName;

            try
                env = gymModule.make(envName);
            catch
                warning(['Failed to load ' char(envName)]);
                continue;
            end

            obsSpace = env.observation_space;
            if isa(obsSpace, 'py.gymnasium.spaces.discrete.Discrete')
                Prob.tasks(i).obsDim = double(obsSpace.n);
            else
                Prob.tasks(i).obsDim = double(obsSpace.shape{1});
            end

            if isa(env.action_space, 'py.gymnasium.spaces.discrete.Discrete')
                Prob.tasks(i).actDim = double(env.action_space.n);
            else
                Prob.tasks(i).actDim = double(env.action_space.shape{1});
            end
            env.close();

            % --- Initialize Welford ---
            oDim = Prob.tasks(i).obsDim;
            Prob.ObsCount{i} = 1e-4;
            Prob.ObsMean{i} = zeros(1, oDim);
            Prob.ObsM2{i} = zeros(1, oDim);
            Prob.ObsStd{i} = ones(1, oDim);

            % --- Dynamic Neural Network Initialization ---
            h = Prob.hiddenSize;
            aDim = Prob.tasks(i).actDim;
            num_layers = Prob.hiddenLayers;

            init_vec = [];
            % 1. Bias 1
            b1 = zeros(1, h);
            init_vec = [init_vec, b1];
            % 2. Middle Layers
            limit_mid = sqrt(3 / h);
            for l = 1:(num_layers - 1)
                W_mid = (rand(1, h * h) * 2 * limit_mid) - limit_mid;
                b_mid = zeros(1, h);
                init_vec = [init_vec, W_mid, b_mid];
            end
            % 3. Output Layer
            limit_out = sqrt(3 / h);
            W_end = (rand(1, h * aDim) * 2 * limit_out) - limit_out;
            b_end = zeros(1, aDim);
            init_vec = [init_vec, W_end, b_end];
            % 4. Input Layer (W1)
            limit_in = sqrt(3 / oDim);
            W1 = (rand(1, oDim * h) * 2 * limit_in) - limit_in;
            init_vec = [init_vec, W1];

            Prob.initSolutionMean{i} = init_vec;
            Prob.tasks(i).paramCount = length(init_vec);
        end

        for i = 1:Prob.T
            Prob.M(i) = 1;
            Prob.D(i) = Prob.tasks(i).paramCount;
            Prob.Lb{i} = -1 * ones(1, Prob.D(i));
            Prob.Ub{i} = 1 * ones(1, Prob.D(i));

            % Boundary is used to control sigma scale
            if i == 3 % Pendulum-v1 sigma=1
                Prob.initSigmaScale{i} = 1/0.3/2;
            elseif i <= 7 % Classic Control & Box2D sigma=0.5
                Prob.initSigmaScale{i} = 0.5/0.3/2;
            elseif i == 16 % Ant-v5 sigma=0.1
                Prob.initSigmaScale{i} = 0.1/0.3/2;
            else % Others Mujoco sigma=0.3
                Prob.initSigmaScale{i} = 0.3/0.3/2;
            end
        end
    end

    function [Obj, Con] = evaluate(Prob, x, t)
        dim = Prob.D(t);
        if any(isnan(x(:, 1:dim)))
            Obj = inf(size(x, 1), 1); Con = zeros(size(x, 1), 1); return;
        end

        % --- Generate Seed in MATLAB ---
        % Generates a large random integer to seed Python's RNG.
        % This ensures MATLAB controls the stochasticity.
        currentSeed = int32(randi(2^31 - 1));

        % Pass Seed to Python
        [Obj, batch_n, batch_mean, batch_m2] = TaskWrapper(t, x, dim, ...
            Prob.hiddenSize, Prob.hiddenLayers, Prob.numRollouts, ...
            Prob.ObsMean{t}, Prob.ObsStd{t}, currentSeed);

        Con = zeros(size(x, 1), 1);

        % Update Welford Statistics
        if batch_n > 0
            n_a = Prob.ObsCount{t};
            mean_a = Prob.ObsMean{t};
            m2_a = Prob.ObsM2{t};

            n_b = batch_n;
            mean_b = batch_mean;
            m2_b = batch_m2;

            n_new = n_a + n_b;
            delta = mean_b - mean_a;

            mean_new = mean_a + delta * (n_b / n_new);
            m2_new = m2_a + m2_b + (delta.^2) * (n_a * n_b / n_new);

            Prob.ObsCount{t} = n_new;
            Prob.ObsMean{t} = mean_new;
            Prob.ObsM2{t} = m2_new;

            variance = m2_new ./ max(1, n_new - 1);
            Prob.ObsStd{t} = sqrt(variance +1e-8);
        end
    end
end
end

function [Obj, batch_n, batch_mean, batch_m2] = TaskWrapper(taskIdx, x, dim, hiddenSize, hiddenLayers, numRollouts, meanVec, stdVec, seed)
% TaskWrapper: Passes parameters AND seed to Python.

params_matrix = py.numpy.array(x(:, 1:dim), dtype = py.numpy.float32);
mean_py = py.numpy.array(meanVec, dtype = py.numpy.float32);
std_py = py.numpy.array(stdVec, dtype = py.numpy.float32);

% Call Python with Seed
result_list = py.gym_runner.run_episode( ...
    int32(taskIdx - 1), ...
    params_matrix, ...
    int32(hiddenSize), ...
    int32(numRollouts), ...
    mean_py, ...
    std_py, ...
    int32(hiddenLayers), ...
    int32(seed) ... % Pass the seed
);

py_data = cell(result_list);

scores = double(py_data{1});
Obj = -scores';

batch_n = double(py_data{2});
batch_mean = double(py_data{3});
batch_m2 = double(py_data{4});
end
