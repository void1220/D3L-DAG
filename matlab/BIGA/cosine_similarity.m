% Legacy MATLAB scheduling routine.
function sim = cosine_similarity(x, y)
    sim = sum(x .* y, 3)./ (sqrt(sum(x.^2, 3)) .* sqrt(sum(y.^2, 3)));
end
