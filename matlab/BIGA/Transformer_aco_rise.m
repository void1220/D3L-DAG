% Legacy MATLAB scheduling routine.
function poptemp = Transformer_aco_rise(pop,info,data,aco_net,len,dlX,dlZ)
pop_size = size(pop,1);
popnew = [];
for i = 1 : pop_size
    state = struct;
    state.HiddenState = dlZ(:,i);
    state.CellState = zeros(size(dlZ(:,i)),'like',dlZ(:,i));
    temp_dlx = dlX(i,:,:);
    dlY = modelDecoder1(aco_net, temp_dlx, state);
    [~,idx] = max(dlY,[],1);
    idx = squeeze(idx)';
    res = idx(1:end-1);
    solnew = res-1;
    for k=1 : data.n
        if (solnew(k) > data.m)||(solnew(k)<1)
            solnew(k) = randi([1 data.m]);
        end
    end
    sol = [solnew, pop(i,len+1:end)];
    popnew = [popnew; sol];
end
poptemp = zeros(pop_size,size(popnew(1,:),2));
for i = 1 : pop_size
    for j = 1:size(popnew(1,:),2)
        poptemp(i,j) = popnew(i,j);
    end
end
end
