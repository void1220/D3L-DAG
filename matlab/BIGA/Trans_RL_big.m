% Legacy MATLAB scheduling routine.
function [energy, time] = Trans_RL_big(info, data)
if info.mode == 1
    base = 'fft';
else
    base = 'gs';
end
agentname = sprintf('AC_%s%d.mat', base, info.num);
loadedAgent = load(agentname, 'agent');
seqFilename = sprintf('%s%d_transformer_rl_seq.mat', base, info.num);
load(seqFilename);
[info,sol,~] = heft_sol(info,data);
info.fitbest=decode(sol,info,data);
info.solbest=sol;
agent = loadedAgent.agent;
ActorNet = getModel(getActor(agent));
fit_curve = zeros(1, info.ng);
curve_std = zeros(1, info.ng);
n=info.tl;
action_sum = zeros(n,20);
tic
min_fit_all = zeros(1,n);
for x = 1 : n
    pop = rand(info.np, data.n*3);
    fit = decode_01(pop, info,data);
    [pop_size,~] = size(pop);
    if info.num<4
        num_LD = pop_size / 2;
    elseif info.num<8
        num_LD = pop_size / 4;
    elseif info.num<10
        num_LD = ceil(pop_size / 7);
    elseif info.num<=16
        num_LD = pop_size / 10;
    end
    num_HD = pop_size - num_LD;
    fit_HD = fit(1:num_HD);
    fit_LD = fit(num_LD+1 : pop_size);
    min_fit = inf;
    fitold = fit;
    num_de_fit = 0;
    for i = 1 : info.ng
        time_cost = toc;
        if time_cost > info.maxrt * 0.8
            break
        end
        de_fit = ( min(fitold) - min(fit) ) / min(fitold);
        if de_fit == 0
            num_de_fit = num_de_fit + 1;
        else
            num_de_fit = 0;
        end
        [inputFeatures, fit] = calculate_input(pop, info, data, fit, fitold);
        result = ActorNet.predict(inputFeatures);
        action = cell2mat( getAction(agent, {inputFeatures}) );
        fitold = fit;
        switchgear = 1;
        if ( ( num_de_fit == 20 && rand()<0.5 ) || i == 1 ) && switchgear == 1
            [pop_HD, pop_LD] = Divide_pop(pop, info, data, num_LD, num_HD);
            fit_HD = fit(1 : num_HD);
            fit_LD = fit(num_HD+1 : pop_size);
            len_LD = size(pop_LD,1);
            pop_LD_aco = pop_LD(1:ceil(len_LD/2),:);
            fit_LD_aco = fit_LD(1:ceil(len_LD/2));
            pop_LD_seq = pop_LD(ceil(len_LD/2)+1:len_LD,:);
            fit_LD_seq = fit_LD(ceil(len_LD/2)+1:len_LD);
            [pop_LD_seq, fit_LD_seq] = Transformer_seq_LDOA(pop_LD_seq, data, info, seq_net, fit_LD_seq);
            pop_LD = [pop_LD_aco; pop_LD_seq];
            fit_LD = [fit_LD_aco, fit_LD_seq];
            [info,pop_HD,fit_HD] = RL_action(pop_HD,data,info,action,fit_HD);
            pop = [pop_HD;pop_LD];
            fit = [fit_HD,fit_LD];
            num_de_fit = 0;
        else
            action_sum(x,action) = action_sum(x,action) + 1;
            [info,pop,fit] = RL_action(pop,data,info,action,fit);
        end
        fit_curve(i) = min(fit);
        curve_std(i) = std(fit);
        min_fitnew = min(fit);
        min_fit = min(min_fitnew, min_fit);
    end
    min_fit_all(x) = min_fit;
    if time_cost>=info.maxrt
        break
    end
end
valid_values = min_fit_all(~isinf(min_fit_all));
energy=min(valid_values);
time=time_cost;
end
