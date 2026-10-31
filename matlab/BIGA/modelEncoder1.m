% Legacy MATLAB scheduling routine.
function dlZ = modelEncoder1(encoderEmbeddingsArray, decoderEmbeddingsArray, params,numBlocks,d_k, d_v,numHeads,   pad_idx,encoderInputArray,decoderInputArray)
encoderOutput = encoderEmbeddingsArray;
similarity_threshold = 0.90;
for i = 1:numBlocks
    encoderOutput = transformerEncoderLayer(encoderOutput, pad_idx,d_k, d_v, numHeads,params, encoderInputArray);
    if exist('encoderOutput1', 'var')
        sim = cosine_similarity(encoderOutput, encoderOutput1);
        if sim > similarity_threshold
            break;
        end
    else
    end
    encoderOutput1 = encoderOutput;
end
decoderOutput = decoderEmbeddingsArray;
for i = 1:numBlocks
    dec_output = DecoderAttention(decoderOutput, d_k, numHeads, params,decoderInputArray,pad_idx);
    dec_output = permute(dec_output, [3,1,2]);
    dec_output = dec_output +decoderOutput;
    dec_output= LayerNorm(dec_output,params);
    decoderOutput = EncoderDecoderAttention(encoderOutput,dec_output , numHeads,pad_idx, d_k, d_v,  params,encoderInputArray);
    decoderOutput = permute(decoderOutput,[3,1,2]);
    decoderOutput = decoderOutput + dec_output;
    decoderOutput = LayerNorm(decoderOutput,params);
    ffc_output = PositionwiseFeedForward2(decoderOutput, params);
    ffc_output = ffc_output +decoderOutput;
    decoderOutput = LayerNorm(ffc_output ,params);
    if exist('decoderOutput1', 'var')
        sim = cosine_similarity(decoderOutput, decoderOutput1);
        if sim > similarity_threshold
            break;
        end
    else
    end
    decoderOutput1 = decoderOutput;
end
[batchSize,~, embeddingDim] = size(decoderOutput);
sequenceLengths = sum(decoderInputArray ~= pad_idx, 2);
globalRepresentation = zeros(batchSize, embeddingDim, 'like', decoderOutput);
for i = 1:batchSize
    t = sequenceLengths(i);
    if t > 0
        globalRepresentation(i,:) = decoderOutput(i, t, :);
    else
        globalRepresentation(i, :) = zeros(1, embeddingDim, 'like', decoderOutput);
    end
end
globalRepresentation = globalRepresentation';
weights = params.fcEncoder.Weights;
bias = params.fcEncoder.Bias;
dlZ = fullyconnect(globalRepresentation,weights,bias,'DataFormat','CB');
end
function output =DecoderAttention(dec_output, d_k, numHeads,params,decoderInput,pad_idx)
[batch_size, seq_len1, ~] = size(dec_output);
seq_len2 = seq_len1;
seq_len3 = seq_len1;
dec_output= permute(dec_output, [2, 3, 1]);
q =  pagemtimes(dec_output ,params.decoder.Wq );
k =  pagemtimes(dec_output ,params.decoder.Wk );
v =  pagemtimes(dec_output ,params.decoder.Wv);
q =  permute(q, [3 1 2]);
k =  permute(k, [3 1 2]);
v =  permute(v, [3 1 2]);
q = reshape(q, [batch_size, seq_len1, numHeads, d_k]);
k = reshape(k, [batch_size, seq_len2, numHeads, d_k]);
v = reshape(v, [batch_size, seq_len3, numHeads, d_k]);
q = permute(q, [2 4 3 1]);
k =  permute(k, [2 4 3 1]);
q = reshape(q, [seq_len1, d_k,batch_size *numHeads]);
k =  reshape(k, [seq_len2, d_k,batch_size *numHeads]);
mask = get_subsequent_mask(decoderInput,pad_idx);
mask = repmat(mask, [numHeads, 1, 1]);
scores =  pagemtimes(q , 'none',k,'transpose') / sqrt(d_k);
scores =  permute(scores, [ 3 1 2]);
scores = scores + (mask * -1e9);
attentionWeights = stable_softmax(scores);
attentionWeights = permute(attentionWeights, [1 3 2]);
attentionWeights=  permute(attentionWeights, [3 2 1]);
v =  permute(v, [2 4 3 1]);
v = reshape(v, [seq_len3, d_k,batch_size *numHeads]);
attention =pagemtimes(attentionWeights , v);
attention =  permute( attention, [3 1 2]);
attention   = reshape(attention, [batch_size, numHeads, seq_len1, d_k]);
attention  =  permute( attention , [1 3 2 4]);
attention = reshape(attention, [batch_size, seq_len1,numHeads * d_k]);
attention =  permute( attention, [2 3 1]);
output =  pagemtimes(attention, params.decoder.Wo);
end
function output =EncoderDecoderAttention(enc_output,dec_output, numHeads, pad_idx, d_k, d_v, params,encoderInput)
batch_size = size(dec_output, 1);
len_q = size(dec_output, 2);
len_k = size(enc_output, 2);
len_v = size(enc_output, 2);
dec_output =  permute(dec_output, [2,3,1]);
enc_output =  permute(enc_output, [2,3,1]);
q =  pagemtimes(dec_output,params.enc_dec.Wq);
k =  pagemtimes(enc_output,params.enc_dec.Wk);
v =  pagemtimes(enc_output,params.enc_dec.Wv);
q = reshape(q, [batch_size, len_q, numHeads, d_k]);
k = reshape(k, [batch_size, len_k, numHeads, d_k]);
v = reshape(v, [batch_size, len_v, numHeads, d_v]);
q = permute(q, [2, 4, 3, 1]);
k = permute(k, [2, 4, 3, 1]);
v = permute(v, [2, 4, 3, 1]);
q = reshape(q, [len_q, d_k,batch_size *numHeads]);
k =  reshape(k, [len_k, d_k,batch_size *numHeads]);
v = reshape(v, [len_v, d_v,batch_size *numHeads]);
mask = get_pad_mask(encoderInput, pad_idx);
mask = repmat(reshape(mask, [batch_size, 1, len_k]), numHeads, 1, 1);
scores =  pagemtimes(q , 'none',k,'transpose') / sqrt(d_k);
scores =  permute(scores, [3 1 2]);
scores = scores + (mask * -1e9);
attentionWeights = stable_softmax(scores);
attentionWeights=  permute(attentionWeights, [1 3 2]);
attentionWeights =  permute(attentionWeights, [3 2 1]);
attention =pagemtimes(attentionWeights , v);
attention =  permute( attention, [3 1 2]);
attention = reshape(attention, [batch_size,numHeads,len_q , d_v]);
attention =  permute( attention, [1 3 2 4]);
attention = reshape(attention, [batch_size,  len_q ,numHeads * d_v]);
attention =  permute(attention, [2 3 1]);
output =  pagemtimes( attention, params.enc_dec.Wo);
end
function output = PositionwiseFeedForward2(x, params)
W1 = params.enc_dec.W1;
W2 = params.enc_dec.W2;
b1 = params.enc_dec.b1;
b2 = params.enc_dec.b2;
x = permute(x, [2, 3, 1]);
x = relu(pagemtimes(W1 , 'none',x,'transpose') + b1);
x = pagemtimes(W2 ,x)+ b2;
output = permute(x,[3, 2, 1]);
end
function output = relu(x)
output = max(0, x);
end
function mask = get_pad_mask(seq, pad_idx)
mask = logical(seq == pad_idx);
end
function mask = get_subsequent_mask(tra_inp,pad_idx)
[sz_b, len_s] = size(tra_inp);
mask1 = get_pad_mask(tra_inp, pad_idx);
mask1 = reshape(mask1, [sz_b,  1, len_s]);
subsequent_mask = triu(ones(len_s, len_s), 1);
subsequent_mask= reshape(subsequent_mask, [ 1,len_s, len_s]);
mask = mask1|subsequent_mask;
end
function y = stable_softmax(x)
x_max = max(x, [], 3);
x_max = reshape(x_max, [size(x,1), size(x,2), 1]);
x_exp = exp(x - x_max);
sum_exp = sum(x_exp, 3);
sum_exp = reshape(sum_exp, [size(x,1), size(x,2), 1]);
sum_exp(sum_exp == 0) = eps;
y = x_exp ./ sum_exp;
end
function output = LayerNorm(x,params)
epsilon = 1e-6;
mu = mean(x, 3);
sigma = std(x, 0, 3);
output = (x - mu) ./ (sigma + epsilon);
output = output .* params.gamma + params.beta;
end
