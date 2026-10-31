% Legacy MATLAB scheduling routine.
clear
clc
load datain_rl.mat
info.flag=1;
info.mode = 1;
info.num  = 10;
if info.mode==1
    temp=['data=datain_rl.fft',num2str(info.num),';'];
elseif info.mode==2
    temp=['data=datain_rl.gs',num2str(info.num),';'];
else
    error('unavailable');
end
eval(temp)
info = setuppara(info);
[energy, time, fit_curve] = BiGA(info, data);
function [energy, time, fit_curve] = BiGA(info, data)
pop = rand(info.np, data.n * 3);
info.n = data.n;
info.m = data.m;
info.h = 1;
info.cmax = data.cmax;
fit = zeros(1, info.np);
fit_curve = zeros(1, info.ng);
tic;
for gen = 1:info.ng
    time_cost = toc;
    if time_cost > info.maxrt
        break
    end
    for i = 1:info.np
        x_upper = pop(i, :);
        fit(i) = lower_level_ga(x_upper, info, data);
    end
    fit_curve(gen) = min(fit);
    [info, pop, fit] = operator_1_biga(pop, data, info, fit);
end
energy = min(fit);
time = toc;
end
function f_lower = lower_level_ga(x_upper, info, data)
popSize = 10;
genMax = 20;
nvars = data.n;
pop = rand(popSize, nvars);
for gen = 1:genMax
    fit = zeros(1, popSize);
    for i = 1:popSize
        y = pop(i, :);
        fit(i) = lower_objective(x_upper, y);
    end
    [~, idx] = sort(fit);
    pop = pop(idx(1:popSize), :);
    for i = 1:2:popSize-1
        alpha = rand;
        pop(i,:)   = alpha * pop(i,:) + (1-alpha) * pop(i+1,:);
        pop(i+1,:) = alpha * pop(i+1,:) + (1-alpha) * pop(i,:);
    end
    for i = 1:popSize
        if rand < 0.2
            pop(i,:) = pop(i,:) + randn(1, nvars) * 0.05;
            pop(i,:) = max(0, min(1, pop(i,:)));
        end
    end
end
best_idx = find(fit == min(fit), 1);
y_best = pop(best_idx, :);
f_lower = lower_objective(x_upper, y_best);
end
function f = lower_objective(x, y)
f = sum((y - x(1:length(y))).^2);
end
function [info, pop, fit] = operator_1_biga(pop, data, info, fit)
[pop_size, ~] = size(pop);
for x = 1 : info.tl
    [info, pop1] = RL_GA_cross(pop, info, fit);
    [info, popout] = RL_GA_mutation2(pop1, info, data);
    fitnew = zeros(1, pop_size);
    for i = 1:pop_size
        fitnew(i) = lower_level_ga(popout(i, :), info, data);
    end
    for i = 1 : pop_size
        if fitnew(i) < fit(i)
            pop(i,:) = popout(i,:);
            fit(i) = fitnew(i);
        end
    end
end
end
