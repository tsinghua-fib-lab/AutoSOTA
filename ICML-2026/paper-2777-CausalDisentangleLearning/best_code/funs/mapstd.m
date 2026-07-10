function y = mapstd(x, ymean, ystd)
    % mapstd replacement for Octave
    % Normalizes each row of x to have given mean and standard deviation
    % x: input matrix (each row is a variable, each column is an observation)
    % ymean: target mean (default 0)
    % ystd: target standard deviation (default 1)
    if nargin < 2
        ymean = 0;
    end
    if nargin < 3
        ystd = 1;
    end
    mu = mean(x, 2);
    sigma = std(x, 0, 2);
    sigma(sigma == 0) = 1;  % avoid division by zero for constant rows
    y = ymean + ystd .* (x - mu) ./ sigma;
end
