% Legacy MATLAB scheduling routine.
function activated = activationFunction(x)
    activated = 0.5 * x .* (1 + erf_approx(x / sqrt(2)));
end
