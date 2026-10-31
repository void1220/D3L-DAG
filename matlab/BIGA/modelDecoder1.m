% Legacy MATLAB scheduling routine.
function [dlY,state] = modelDecoder1(params,seq,state)
 embeddingMatrix =  params.modelDecoder1.embeddingMatrix;
 embeddedSeq = embedSequences(seq, embeddingMatrix, 100);
 dlX = permute(embeddedSeq, [3,1,2]);
inputWeights = params.lstmDecoder.InputWeights;
recurrentWeights = params.lstmDecoder.RecurrentWeights;
bias = params.lstmDecoder.Bias;
hiddenState = state.HiddenState;
cellState = state.CellState;
[dlY,hiddenState,cellState] = lstm(dlX,hiddenState,cellState, ...
    inputWeights,recurrentWeights,bias,'DataFormat','CBT');
state.HiddenState = hiddenState;
state.CellState = cellState;
weights = params.fcDecoder.Weights;
bias = params.fcDecoder.Bias;
dlY = fullyconnect(dlY,weights,bias,'DataFormat','CBT');
dlY = softmax(dlY,'DataFormat','CBT');
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
