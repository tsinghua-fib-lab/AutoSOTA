function Result = gen2eva_rl(Result_Gen, FE_Gen, maxGen)
%% Map the convergence from generation to evaluation
% Input: Result_Gen, FE_Gen, maxGen
% Output: Result

if size(Result_Gen, 2) <= maxGen
    repeat_factor = ceil((maxGen + 1) / size(Result_Gen, 2));
    Result_Gen = repelem(Result_Gen, 1, repeat_factor);
    FE_Gen = repelem(FE_Gen, 1, repeat_factor);
end

Result = Result_Gen(:, 1:maxGen);
for k = 1:size(Result_Gen, 1)
    Gap = FE_Gen(end) ./ (maxGen);
    idx = 1;
    i = 1;
    while i <= length(FE_Gen)
        if FE_Gen(i) >= ((idx) * Gap)
            Result(k, idx) = Result_Gen(k, i);
            idx = idx + 1;
        else
            i = i + 1;
        end
        if idx > maxGen
            break;
        end
    end
    Result(k, 1) = Result_Gen(k, 1);
    Result(k, end) = Result_Gen(k, end);
    if idx < maxGen
        for x = idx:maxGen
            Result(k, x) = Result_Gen(k, end);
        end
    end
end
end
