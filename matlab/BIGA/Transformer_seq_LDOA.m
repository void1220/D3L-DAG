% Legacy MATLAB scheduling routine.
function [popout, fit] = Transformer_seq_LDOA(pop,data,info,seq_net,fit)
[pop_size,~] = size(pop);
[pop1,len_LSTM_seq,dlX_all,dlZ_all] = Transformer_seq_reduction(pop,info,data,seq_net);
[info,pop2] = RL_GA_cross(pop1,info,fit);
[info,pop3] = RL_GA_mutation2(pop2, info, data);
popout = Transformer_seq_rise(pop3,info,data,seq_net,len_LSTM_seq,dlX_all,dlZ_all);
for i = 1 : pop_size
    popout(i,1:data.n) = ceil(popout(i,1:data.n) * data.m);
    [info,popout(i,:)]=swapsol_aco(popout(i,:),data,info);
    [info,popout(i,:)]=insert_aco(popout(i,:),data,info);
    [info,popout(i,:)]=mutate_aco(popout(i,:),data,info);
end
buchang_aco = 1 / data.m;
buchang_seq = 1 / data.n;
for i = 1:pop_size
    for j = 1:data.n
        popout(i,j) = popout(i,j) / data.m - rand() * buchang_aco;
        popout(i,data.n + j) = popout(i,data.n + j) / data.n - rand() * buchang_seq;
    end
end
[fitnew,~,~] = decode_01(popout,info,data);
for i = 1 : pop_size
    if fitnew(i) < fit(i)
        pop(i,:) = popout(i,:);
        fit(i)=fitnew(i);
    end
end
end
