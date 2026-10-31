% Legacy MATLAB scheduling routine.
function dlY = modelDecoderPredictions(parameters,dlZ,maxLength,enc,startToken,miniBatchSize)
numObservations = size(dlZ,2);
numIterations = ceil(numObservations / miniBatchSize);
startTokenIdx = word2ind(enc,startToken);
vocabularySize = enc.NumWords;
dlY = zeros(vocabularySize,numObservations,maxLength,'like',dlZ);
for i = 1:numIterations
    idxMiniBatch = (i-1)*miniBatchSize+1:min(i*miniBatchSize,numObservations);
    miniBatchSize = numel(idxMiniBatch);
    state = struct;
    state.HiddenState = dlZ(:,idxMiniBatch);
    state.CellState = zeros(size(dlZ(:,idxMiniBatch)),'like',dlZ);
    decoderInput = dlarray(repmat(startTokenIdx,[1 miniBatchSize]),'CBT');
    for t = 1:maxLength
        [dlY(:,idxMiniBatch,t), state] = modelDecoder(parameters,decoderInput,state);
        [~,idx] = max(dlY(:,idxMiniBatch,t));
        decoderInput = idx;
    end
end
end
