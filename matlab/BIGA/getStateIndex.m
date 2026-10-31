% Legacy MATLAB scheduling routine.
function stateIndex = getStateIndex(stateValues)
    n1 = 10;
    n2 = 5;
    range1 = [0, 1];
    range2 = [0, 1];
    i1 = nonLinearDiscretize(stateValues(1), range1, n1);
    i2 = nonLinearDiscretize(stateValues(2), range2, n2);
    stateIndex = i1 * n2 + i2 + 1;
end
function index = nonLinearDiscretize(value, range, numBins)
    value = max(min(value, range(2)), range(1));
    if value > 0.1
        index = floor((log10(value / 0.1) / log10(range(2) / 0.1)) * (numBins * 0.3));
    elseif value > 0.01
        index = floor((log10(value / 0.01) / log10(0.1 / 0.01)) * (numBins * 0.4)) + floor(numBins * 0.3);
    elseif value > 0.001
        index = floor((log10(value / 0.001) / log10(0.01 / 0.001)) * (numBins * 0.2)) + floor(numBins * 0.7);
    else
        index = floor((value / 0.001) * (numBins * 0.1)) + floor(numBins * 0.9);
    end
    index = min(max(index, 0), numBins - 1);
end
