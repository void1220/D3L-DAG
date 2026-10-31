% Legacy MATLAB scheduling routine.
function oh = oneHot(idx, outputSize)
miniBatchSize = numel(idx);
oh = zeros(outputSize,miniBatchSize);
for n = 1:miniBatchSize
    c = idx(n);
    oh(c,n) = 1;
end
end
