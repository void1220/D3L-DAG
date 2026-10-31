% Legacy MATLAB scheduling routine.
function y = erf_approx(x)
    sign_x = sign(x);
    x = abs(x);
    a1 = 0.254829592;
    a2 = -0.284496736;
    a3 = 1.421413741;
    a4 = -1.453152027;
    a5 = 1.061405429;
    p = 0.3275911;
    t = 1 ./ (1 + p .* x);
    poly = ((a5 .* t + a4) .* t + a3) .* t + a2;
    poly = (poly .* t + a1) .* t;
    y = 1 - poly .* exp(-x.^2);
    y = sign_x .* y;
end
