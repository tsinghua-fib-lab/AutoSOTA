classdef MaT_CEC20_RWCO < Problem
% <Many-task> <Single-objective> <Constrained>

methods
    function Prob = MaT_CEC20_RWCO(varargin)
        Prob = Prob@Problem(varargin);
        Prob.maxFE = 42 * 100 * 500;
    end

    function setTasks(Prob)
        Prob.T = 42;
        for i = 1:Prob.T
            if i <= 5 % RC1-5
                Tasks(i) = benchmark_CEC20_RWCO(i);
            elseif i <= 31 % RC8-33
                Tasks(i) = benchmark_CEC20_RWCO(i + 2);
            else % RC44-54
                Tasks(i) = benchmark_CEC20_RWCO(i + 12);
            end
            Prob.D(i) = Tasks(i).Dim;
            Prob.Fnc{i} = Tasks(i).Fnc;
            Prob.Lb{i} = Tasks(i).Lb;
            Prob.Ub{i} = Tasks(i).Ub;
        end
    end
end
end
