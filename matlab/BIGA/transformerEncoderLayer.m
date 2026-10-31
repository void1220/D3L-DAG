% Legacy MATLAB scheduling routine.
function enc_output  =transformerEncoderLayer(encoderOutput, pad_idx,d_k, d_v, numHeads,params, encoderInputArray)
    enc_output = MultiHeadAttention(encoderOutput,pad_idx, d_k, d_v, numHeads,params,encoderInputArray);
    enc_output = permute(enc_output, [3,1,2]);
    enc_output = enc_output +encoderOutput;
   enc_output = LayerNorm(enc_output,params);
    ffc_output = PositionwiseFeedForward(enc_output,params);
    ffc_output = ffc_output + enc_output;
    enc_output = LayerNorm(ffc_output,params);
end
function output =MultiHeadAttention(embeddedInputData,pad_idx, d_k, d_v, numHeads,params,encoderInput)
    [batch_size, seq_len1, ~] = size(embeddedInputData);
    seq_len2 = seq_len1;
    seq_len3 = seq_len1;
      embeddedInputData= permute(embeddedInputData, [2, 3, 1]);
    q =  pagemtimes(embeddedInputData , params.encoder.Wq);
    k =  pagemtimes(embeddedInputData , params.encoder.Wk);
    v =  pagemtimes(embeddedInputData ,params.encoder.Wv);
     q =  permute(q, [3 1 2]);
     k =  permute(k, [3 1 2]);
     v =  permute(v, [3 1 2]);
    q = reshape(q, [batch_size, seq_len1, numHeads, d_k]);
    k = reshape(k, [batch_size, seq_len2, numHeads, d_k]);
    v = reshape(v, [batch_size, seq_len3, numHeads, d_v]);
    q = permute(q, [2 4 3 1]);
    k =  permute(k, [2 4 3 1]);
    q = reshape(q, [seq_len1, d_k,batch_size *numHeads]);
    k =  reshape(k, [seq_len2, d_k,batch_size *numHeads]);
     mask = get_pad_mask(encoderInput, pad_idx);
   mask = reshape(mask, [batch_size, 1, seq_len2]);
   mask = repmat(mask, [numHeads,1, 1]);
    attention_scores =  pagemtimes(q , 'none',k,'transpose') / sqrt(d_k);
    attention_scores =  permute(attention_scores, [3 1 2]);
     attention_scores = attention_scores + (mask * -1e9);
    attention_probs = stable_softmax(attention_scores);
     attention_probs=  permute(attention_probs, [1 3 2]);
    attention_probs=  permute(attention_probs, [3 2 1]);
    v =  permute(v, [2 4 3 1]);
    v = reshape(v, [seq_len3, d_v,batch_size *numHeads]);
    Oattention =pagemtimes(attention_probs , v);
    Oattention =  permute(Oattention, [ 3 1 2]);
    Oattention  = reshape(Oattention, [batch_size, numHeads, seq_len1, d_v]);
    Oattention =  permute( Oattention, [1 3 2 4]);
    Oattention = reshape(Oattention, [batch_size, seq_len1,numHeads * d_v]);
    Oattention =  permute(Oattention, [2 3 1]);
    output =  pagemtimes( Oattention, params.encoder.Wo);
end
function mask = get_pad_mask(seq, pad_idx)
    mask = logical(seq == pad_idx);
end
function output = LayerNorm(x,params)
    epsilon = 1e-6;
    mu = mean(x, 3);
    sigma = std(x, 0, 3);
    output = (x - mu) ./ (sigma + epsilon);
output = output .* params.gamma + params.beta;
end
function output = PositionwiseFeedForward(x,params)
    W1 = params.encoder.W1;
    W2 = params.encoder.W2;
    b1 = params.encoder.b1;
    b2 = params.encoder.b2;
     x = permute(x, [2, 3, 1]);
     x =activationFunction(pagemtimes(W1 , 'none',x,'transpose') + b1);
     x = pagemtimes(W2 ,x)+ b2;
     output  = permute(x,[3, 2, 1]);
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
