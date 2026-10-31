% Legacy MATLAB scheduling routine.
function [encoderEmbeddingsArray, decoderEmbeddingsArray, encoderInputArray, decoderInputArray,decoderTargetArray] = cellToTrainingArrays(encoderEmbeddings, decoderEmbeddings, encoderInput, decoderInput, decoderTarget, pad_idx)
maxEncoderSeqLen = max(cellfun(@(x) size(x, 2), encoderEmbeddings));
maxDecoderSeqLen = max(cellfun(@(x) size(x, 2), decoderEmbeddings));
encoderEmbeddingsArray = [];
decoderEmbeddingsArray = [];
encoderInputArray = [];
decoderInputArray = [];
decoderTargetArray = [];
for i = 1:length(encoderEmbeddings)
    batchEncoderEmbeddings = encoderEmbeddings{i};
    batchDecoderEmbeddings = decoderEmbeddings{i};
    batchEncoderInput = encoderInput{i};
    batchDecoderInput = decoderInput{i};
    batchDecoderTarget = decoderTarget{i};
    [batchSize, encoderSeqLen, embeddingDim] = size(batchEncoderEmbeddings);
    [~, decoderSeqLen, ~] = size(batchDecoderEmbeddings);
    paddedEncoderEmbeddings = padarray(batchEncoderEmbeddings, [0, maxEncoderSeqLen - encoderSeqLen, 0], pad_idx, 'post');
    paddedDecoderEmbeddings = padarray(batchDecoderEmbeddings, [0, maxDecoderSeqLen - decoderSeqLen, 0], pad_idx, 'post');
    paddedEncoderInput = padarray(batchEncoderInput, [0, maxEncoderSeqLen - size(batchEncoderInput, 2)], pad_idx, 'post');
    paddedDecoderInput = padarray(batchDecoderInput, [0, maxDecoderSeqLen - size(batchDecoderInput, 2)], pad_idx, 'post');
    paddedDecoderTarget = padarray(batchDecoderTarget, [0, maxDecoderSeqLen - size(batchDecoderTarget, 2)], pad_idx, 'post');
    encoderEmbeddingsArray = cat(1, encoderEmbeddingsArray, paddedEncoderEmbeddings);
    decoderEmbeddingsArray = cat(1, decoderEmbeddingsArray, paddedDecoderEmbeddings);
    encoderInputArray = cat(1, encoderInputArray, paddedEncoderInput);
    decoderInputArray = cat(1, decoderInputArray, paddedDecoderInput);
    decoderTargetArray = cat(1, decoderTargetArray, paddedDecoderTarget);
end
end
