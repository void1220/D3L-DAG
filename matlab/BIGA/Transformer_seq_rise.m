% Legacy MATLAB scheduling routine.
function poptemp = Transformer_seq_rise(pop,info,data,seq_net,len,dlX,dlZ)
pop_size = size(pop,1);
popnew = [];
for i = 1 : pop_size
    state = struct;
    state.HiddenState = dlZ(:,i);
    state.CellState = zeros(size(dlZ(:,i)),'like',dlZ(:,i));
    temp_dlx = dlX(i,:,:);
    dlY = modelDecoder1(seq_net, temp_dlx, state);
    [~,idx] = max(dlY,[],1);
    idx = squeeze(idx)';
    res = idx(1:end-1);
    solnew = res-1;
    sol = [ pop(i, 1:data.n), solnew, pop(i, data.n+len+1:end) ];
    popnew = [popnew; sol];
end
poptemp = zeros(pop_size,size(popnew(1,:),2));
for i = 1 : pop_size
    for j = 1:size(popnew(1,:),2)
        poptemp(i,j) = popnew(i,j);
    end
end
end
