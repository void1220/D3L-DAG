% Legacy MATLAB scheduling routine.
function Z = embedding(X, weights)
[N, T] = size(X, 2:3);
X = reshape(X, N*T, 1);
Z = weights(:, X);
Z = reshape(Z, [], N, T);
end
