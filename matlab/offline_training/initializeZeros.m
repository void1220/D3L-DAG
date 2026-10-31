% Legacy MATLAB scheduling routine.
function parameter = initializeZeros(sz)
parameter = zeros(sz,'single');
parameter = dlarray(parameter);
end
