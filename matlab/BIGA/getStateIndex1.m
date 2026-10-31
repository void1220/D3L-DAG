% Legacy MATLAB scheduling routine.
function stateIndex = getStateIndex(stateValues)
    n1 = 5;
    n2 = 5;
    range1 = [0, 1];
    range2 = [0, 1];
    i1 = nonLinearDiscretizeI1(stateValues(1), range1, n1);
    i2 = nonLinearDiscretizeI2(stateValues(2), range2, n2);
    stateIndex = i1 * n2 + i2 + 1;
end
function index = nonLinearDiscretizeI1(value, range, numBins)
    value = max(min(value, range(2)), range(1));
    if value > 0.5
        index = floor(((value - 0.5) / 0.5) * (numBins * 0.05)) + floor(numBins * 0.95);
    elseif value > 0.4
        index = floor(((value - 0.4) / 0.1) * (numBins * 0.05)) + floor(numBins * 0.9);
    elseif value > 0.3
        index = floor(((value - 0.3) / 0.1) * (numBins * 0.05)) + floor(numBins * 0.85);
    elseif value > 0.2
        index = floor(((value - 0.2) / 0.1) * (numBins * 0.05)) + floor(numBins * 0.8);
    elseif value > 0.1
        index = floor(((value - 0.1) / 0.1) * (numBins * 0.1)) + floor(numBins * 0.7);
    elseif value > 0.01
        index = floor(((value - 0.01) / 0.09) * (numBins * 0.1)) + floor(numBins * 0.6);
    elseif value > 0.001
        index = floor(((value - 0.001) / 0.009) * (numBins * 0.2)) + floor(numBins * 0.4);
    elseif value > 0.0001
        index = floor(((value - 0.0001) / 0.0009) * (numBins * 0.2)) + floor(numBins * 0.2);
    else
        index = floor((value / 0.0001) * (numBins * 0.2));
    end
    index = min(max(index, 0), numBins - 1);
end
function index = nonLinearDiscretizeI2(value, range, numBins)
    value = max(min(value, range(2)), range(1));
    if value > 0.9
        index = floor(((value - 0.9) / 0.1) * (numBins * 0.2)) + floor(numBins * 0.8);
    elseif value > 0.8
        index = floor(((value - 0.8) / 0.1) * (numBins * 0.2)) + floor(numBins * 0.6);
    elseif value > 0.7
        index = floor(((value - 0.7) / 0.1) * (numBins * 0.2)) + floor(numBins * 0.4);
    elseif value > 0.6
        index = floor(((value - 0.6) / 0.1) * (numBins * 0.15)) + floor(numBins * 0.25);
    elseif value > 0.5
        index = floor(((value - 0.5) / 0.1) * (numBins * 0.05)) + floor(numBins * 0.2);
    elseif value > 0.4
        index = floor(((value - 0.4) / 0.1) * (numBins * 0.05)) + floor(numBins * 0.15);
    elseif value > 0.3
        index = floor(((value - 0.3) / 0.1) * (numBins * 0.05)) + floor(numBins * 0.1);
    elseif value > 0.2
        index = floor(((value - 0.2) / 0.1) * (numBins * 0.05)) + floor(numBins * 0.05);
    else
        index = floor((value / 0.2) * (numBins * 0.05));
    end
    index = min(max(index, 0), numBins - 1);
end
