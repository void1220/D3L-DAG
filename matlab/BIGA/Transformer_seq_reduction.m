% Legacy MATLAB scheduling routine.
function [poptemp,len,dlX_all,dlZ_all] = Transformer_seq_reduction(pop,info,data,seq_net)
popnew = [];
pop_size = size(pop, 1);
dlX_all=[];
dlZ_all=[];
for i = 1:pop_size
    [~,pop(i,data.n+1:data.n*2)] = sort(pop(i,data.n+1:data.n*2) * data.n);
end
for i = 1:pop_size
    test_data = pop(i,data.n+1:data.n*2);
    params = seq_net;
    batchSize = 1;
    embeddingDim = 100;
    pad_idx = 0;
    numBlocks = 12;
    d_k = 25;
    d_v = 25;
    numHeads = 4;
    [encoderInput, decoderInput, decoderTarget, encoderEmbeddings, decoderEmbeddings] = preprocessSeqTrain(test_data, batchSize, embeddingDim,params);
    [encoderEmbeddings, decoderEmbeddings] = addPositionalEncoding2(encoderEmbeddings, decoderEmbeddings);
    [encoderEmbeddingsArray, decoderEmbeddingsArray, encoderInputArray, decoderInputArray,decoderTargetArray] = cellToTrainingArrays(encoderEmbeddings, decoderEmbeddings, encoderInput, decoderInput, decoderTarget, pad_idx);
    dlZ = modelEncoder1(encoderEmbeddingsArray, decoderEmbeddingsArray, params,numBlocks,d_k, d_v,numHeads, pad_idx,encoderInputArray,decoderInputArray);
    dlX_all = [dlX_all; decoderInputArray];
    lstm_sol = double(dlZ);
    soltemp = lstm_sol';
    dlZ_all = [dlZ_all, lstm_sol];
    popnew = [popnew; pop(i, 1:data.n) soltemp pop(i, data.n*2+1:data.n*3)] ;
end
poptemp = zeros(pop_size, size(popnew(1,:),2));
for i = 1 : pop_size
    for j = 1:size(popnew(1,:),2)
        poptemp(i,j) = popnew(i,j);
    end
end
len = length(soltemp);
end
