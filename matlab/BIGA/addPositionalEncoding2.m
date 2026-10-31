% Legacy MATLAB scheduling routine.
function [encoderEmbeddings, decoderEmbeddings] = addPositionalEncoding2(encoderEmbeddings, decoderEmbeddings)
for batchIdx = 1:length(encoderEmbeddings)
    batchEncoderEmbeddings = encoderEmbeddings{batchIdx};
    batchDecoderEmbeddings = decoderEmbeddings{batchIdx};
    [batchSize1, seqLen1, embeddingDim1] = size(batchEncoderEmbeddings);
    [batchSize2, seqLen2, embeddingDim2] = size(batchDecoderEmbeddings);
    positionEncoding1 = zeros(seqLen1, embeddingDim1);
    for pos = 0:seqLen1-1
        for i = 0:embeddingDim1/2-1
            positionEncoding1(pos+1, 2*i+1) = sin(pos / (10000^(2*i/embeddingDim1)));
            positionEncoding1(pos+1, 2*i+2) = cos(pos / (10000^(2*i/embeddingDim1)));
        end
    end
    positionEncoding1 = reshape(positionEncoding1, [1, seqLen1, embeddingDim1]);
    positionEncoding1 = repmat(positionEncoding1, [batchSize1, 1, 1]);
    if any(isnan(positionEncoding1(:))) || any(isinf(positionEncoding1(:)))
        error('positionEncoding1 包含 NaN 或 Inf 值');
    end
    positionEncoding2 = zeros(seqLen2, embeddingDim2);
    for pos = 0:seqLen2-1
        for i = 0:embeddingDim2/2-1
            positionEncoding2(pos+1, 2*i+1) = sin(pos / (10000^(2*i/embeddingDim2)));
            positionEncoding2(pos+1, 2*i+2) = cos(pos / (10000^(2*i/embeddingDim2)));
        end
    end
    positionEncoding2 = reshape(positionEncoding2, [1, seqLen2, embeddingDim2]);
    positionEncoding2 = repmat(positionEncoding2, [batchSize2, 1, 1]);
    if any(isnan(positionEncoding2(:))) || any(isinf(positionEncoding2(:)))
        error('positionEncoding2 包含 NaN 或 Inf 值');
    end
    encoderEmbeddings{batchIdx} = batchEncoderEmbeddings + positionEncoding1;
    if any(isnan(encoderEmbeddings{batchIdx}(:))) || any(isinf(encoderEmbeddings{batchIdx}(:)))
        error('encoderEmbeddings{%d} 包含 NaN 或 Inf 值', batchIdx);
    end
    decoderEmbeddings{batchIdx} = batchDecoderEmbeddings + positionEncoding2;
    if any(isnan(decoderEmbeddings{batchIdx}(:))) || any(isinf(decoderEmbeddings{batchIdx}(:)))
        error('decoderEmbeddings{%d} 包含 NaN 或 Inf 值', batchIdx);
    end
end
end
