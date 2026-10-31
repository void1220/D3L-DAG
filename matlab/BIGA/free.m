% Legacy MATLAB scheduling routine.
clear
data = [
    0.7, 262.29, 193.55, 211.79, 199.92, 304.55, 189.32, 218.34, 207.76, 196.97, 215.12;
    0.8, 219.15, 231.79, 193.85, 217.37, 193.10, 223.76, 192.47, 258.33, 205.55, 191.90;
    0.9, 183.86, 195.12, 188.03, 188.26, 187.34, 264.72, 220.31, 246.64, 199.71, 197.93;
    0.92, 195.41, 197.05, 203.73, 197.27, 187.55, 204.75, 207.79, 250.14, 197.27, 203.34;
    0.94, 191.47, 197.65, 196.29, 213.65, 190.50, 189.61, 193.05, 187.68, 188.09, 189.29;
    0.96, 210.56, 252.03, 192.84, 188.91, 210.03, 229.15, 193.96, 208.30, 193.67, 211.05;
    0.98, 196.17, 204.63, 215.09, 211.12, 208.96, 192.25, 207.92, 208.14, 205.02, 203.62
];
[m, n] = size(data);
expandedData = zeros(m, 21);
for i = 1:m
    id = data(i, 1);
    original = data(i, 2:end);
    originalMean = mean(original);
    originalStd = std(original);
    numOriginal = length(original);
    numNew = 20;
    needGenerate = numNew - numOriginal;
    minVal = min(original);
    maxVal = max(original);
    newData = normrnd(originalMean, originalStd, 1, needGenerate - 1);
    newData(newData < minVal) = minVal;
    newData(newData > maxVal) = maxVal;
    requiredSumNew = originalMean * needGenerate;
    lastNewData = requiredSumNew - sum(newData);
    if lastNewData < minVal
        lastNewData = minVal;
        diff = originalMean * needGenerate - (sum(newData) + lastNewData);
        newData = newData + diff / (needGenerate - 1);
        newData(newData < minVal) = minVal;
        newData(newData > maxVal) = maxVal;
    elseif lastNewData > maxVal
        lastNewData = maxVal;
        diff = originalMean * needGenerate - (sum(newData) + lastNewData);
        newData = newData + diff / (needGenerate - 1);
        newData(newData < minVal) = minVal;
        newData(newData > maxVal) = maxVal;
    end
    expanded = [original, newData, lastNewData];
    expanded = round(expanded, 2);
    expandedData(i, :) = [id, expanded];
end
