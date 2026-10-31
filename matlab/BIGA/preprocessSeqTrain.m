% Legacy MATLAB scheduling routine.
function [encoderInput, decoderInput, decoderTarget, encoderEmbeddings, decoderEmbeddings] = preprocessSeqTrain(seq_train, batchSize, embeddingDim,params)
if ~iscell(seq_train)
    seq_train = num2cell(seq_train, 2);
end
maxTaskIndex = max(cellfun(@(x) max(x, [], 'omitnan'), seq_train));
padToken = 0;
startToken =1 + maxTaskIndex;
endToken = 2 + maxTaskIndex;
embeddingMatrix =  params.embeddingMatrix;
encoderInput = {};
decoderInput = {};
decoderTarget = {};
encoderEmbeddings = {};
decoderEmbeddings = {};
numSamples = length(seq_train);
numBatches = ceil(numSamples / batchSize);
for batchIdx = 1:numBatches
    batchStartIdx = (batchIdx - 1) * batchSize + 1;
    batchEndIdx = min(batchIdx * batchSize, numSamples);
    currentBatchSize = batchEndIdx - batchStartIdx + 1;
    batchACOSeq = seq_train(batchStartIdx:batchEndIdx);
    [batchEncoderInput, batchDecoderInput, batchDecoderTarget] = prepareInputsTargets(batchACOSeq, startToken, endToken, padToken);
    batchEncoderEmbedding = embedSequences(batchEncoderInput, embeddingMatrix, embeddingDim);
    batchDecoderEmbedding = embedSequences(batchDecoderInput, embeddingMatrix, embeddingDim);
    encoderInput{end+1} = batchEncoderInput;
    decoderInput{end+1} = batchDecoderInput;
    decoderTarget{end+1} = batchDecoderTarget;
    encoderEmbeddings{end+1} = batchEncoderEmbedding;
    decoderEmbeddings{end+1} = batchDecoderEmbedding;
end
end
function [batchEncoderInput, batchDecoderInput, batchDecoderTarget] = prepareInputsTargets(batchACOSeq, startToken, endToken, padToken)
batchSize = length(batchACOSeq);
maxLen = max(cellfun(@length, batchACOSeq));
batchEncoderInput = cell2mat(cellfun(@(x) padSequence([startToken, x, endToken], maxLen + 2, padToken), batchACOSeq, 'UniformOutput', false));
batchDecoderInput = cell2mat(cellfun(@(x) padSequence([startToken, x], maxLen + 1, padToken), batchACOSeq, 'UniformOutput', false));
batchDecoderTarget = cell2mat(cellfun(@(x) padSequence([x, endToken], maxLen + 1, padToken), batchACOSeq, 'UniformOutput', false));
end
function embeddedSeq = embedSequences(seq, embeddingMatrix, embeddingDim)
    [batchSize, seqLength] = size(seq);
    embeddedSeq = zeros(batchSize, seqLength, embeddingDim);
    for i = 1:batchSize
        for j = 1:seqLength
            embeddedSeq(i, j, :) = embeddingMatrix(seq(i, j) , :);
        end
    end
end
function paddedSeq = padSequence(seq, maxLen, padToken)
paddedSeq = [seq, repmat(padToken, 1, maxLen - length(seq))];
end
