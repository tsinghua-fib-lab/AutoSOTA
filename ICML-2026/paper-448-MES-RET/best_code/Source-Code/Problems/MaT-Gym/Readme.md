# MaT-Gym: Many-Task Gymnasium Benchmark

**MaT-Gym** is designed for **Many-Task Reinforcement Learning (RL)** and **Evolutionary Multitasking** research. It bridges the numerical optimization capabilities of MATLAB with the extensive environment suite of Python's `gymnasium`.

## Key Features

* **18 Diverse Tasks**: Includes Classic Control, Box2D, and MuJoCo environments (e.g., `Ant-v5`, `Humanoid-v5`, `BipedalWalker-v3`).
* **Dynamic MLP Architecture**: The neural network policy depth and width are configurable via MATLAB hyperparameters.
* **Numerical Stability**: Implements **Welford’s Online Algorithm** for real-time observation normalization (z-score).
* **Controlled Randomness**: Random seeds are generated in MATLAB and passed to Python to ensure experimental reproducibility.
* **Vectorized Execution**: Supports `SyncVectorEnv` for high-throughput parallel evaluation of populations.

## Installation & Requirements

### Python Environment

1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### MATLAB Configuration (`pyenv`)

To allow MATLAB to call the Python scripts, you must configure the Python interpreter within MATLAB.

1. **Locate your Python Executable**:
   * Windows: `where python`
   * Linux/macOS: `which python`


2. **Set the Python Environment in MATLAB**:
    Run the following commands in the MATLAB Command Window:
    ```matlab
    % Replace with your actual path
    py_path = 'C:\Path\To\Your\Python\python.exe'; 
    % Configure pyenv
    pyenv('Version', py_path);
    % Verify configuration
    pe = pyenv;
    fprintf('Using Python version %s\n', pe.Version);
    ```

### Save and Render Policies

Open `Algorithm.m`, locate the `notTerminated` method, and append the following code block at the very end (before `Algo.Check_Status_Fn()`):

```matlab
% -------------------------------------------------------------------------
% Save the best solution (Decision Variables & Normalizer Stats)
% -------------------------------------------------------------------------
if Algo.FE >= Prob.maxFE && (contains(class(Prob), 'MaT_Gym'))
    % 1. Generate unique timestamp
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');

    % 2. Prepare Dec Data
    Dec = nan(Prob.T, max(Prob.D));
    for t = 1:Prob.T
        % Restore real parameter values from [0,1] normalization
        % Assuming the algorithm optimizes in [0,1] and Prob.Lb/Ub define the real bounds
        real_dec = Algo.Best{t}.Dec(1:Prob.D(t)) .* (Prob.Ub{t} - Prob.Lb{t}) + Prob.Lb{t};
        Dec(t, 1:Prob.D(t)) = real_dec;
    end

    % 3. Prepare Normalizer Data
    normalizer = cell(1, Prob.T);
    for t = 1:Prob.T
        if contains(class(Prob), 'MaT_Gym')
            env_name = char(Prob.tasks(t).name);
        elseif contains(class(Prob), 'MaT_Brax')
            env_name = Prob.envNames{t};
        end
        normalizer{t}.name = env_name;
        normalizer{t}.mean = double(Prob.ObsMean{t});
        normalizer{t}.std = double(Prob.ObsStd{t});
        % Save architecture configuration for reconstruction
        normalizer{t}.hiddenLayers = double(Prob.hiddenLayers);
        normalizer{t}.hiddenSize = double(Prob.hiddenSize);
    end

    % 4. Save Files (Filename format: AlgoName_Type_Timestamp.mat)
    % Replace spaces with underscores to avoid path issues
    algoNameSafe = strrep(Algo.Name, ' ', '_');

    % Define the target directory path
    probClassPath = which(class(Prob));
    if isempty(probClassPath)
        baseDir = '.';
    else
        [baseDir, ~, ~] = fileparts(probClassPath);
    end

    % Construct path: ProbFolder/Data/AlgoName
    targetDir = fullfile(baseDir, 'Data', algoNameSafe);

    % Check if the directory exists; if not, create it
    if ~exist(targetDir, 'dir')
        mkdir(targetDir);
    end

    decFileName = sprintf('%s/%s_Dec_%s.mat', targetDir, algoNameSafe, timestamp);
    normFileName = sprintf('%s/%s_Normalizer_%s.mat', targetDir, algoNameSafe, timestamp);

    % Perform the save operations
    save(decFileName, 'Dec');
    save(normFileName, 'normalizer');

    fprintf('Results saved:\n  [Dec]  %s\n  [Norm] %s\n', decFileName, normFileName);
end
```
