% Legacy MATLAB scheduling routine.
clear
clc
load datain_rl.mat
iteraction = 5;
for algorithm = 4
    for mode = 1
        for num = 1:16
            info.flag = algorithm;
            info.mode = mode;
            info.num = num;
            if info.mode == 1
                temp = ['data = datain_rl.fft', num2str(info.num), ';'];
                mode_str = "fft";
            elseif info.mode == 2
                temp = ['data = datain_rl.gs', num2str(info.num), ';'];
                mode_str = "gs";
            else
                error('unavailable');
            end
            eval(temp);
            info = setuppara(info);
            idx_row = (info.mode - 1) * 16 + info.num;
            idx_col = info.flag;
            for iter = 1:iteraction
                [e_iter, t_iter] = CCS(info, data);
                res_e(idx_row, idx_col, iter) = e_iter;
                res_t(idx_row, idx_col, iter) = t_iter;
                fprintf('%s%d Iteration %d: res_e: %.4f, res_t: %.4f\n', mode_str, info.num, iter, e_iter, t_iter);
            end
            avg_res_e(idx_row, idx_col) = mean(res_e(idx_row, idx_col, :));
            avg_res_t(idx_row, idx_col) = mean(res_t(idx_row, idx_col, :));
            fprintf('%s%d Average res_e: %.4f, Average res_t: %.4f\n', mode_str, info.num, avg_res_e(idx_row, idx_col), avg_res_t(idx_row, idx_col));
        end
    end
end
function [energy, time, fit_curve] = CCS(info, data)
pop = rand(info.np, data.n * 3);
fit = zeros(1, info.np);
info.n = data.n;
info.m = data.m;
info.cmax = data.cmax;
fit_curve = zeros(1, info.ng);
info.gamma = 0.35;
info.pa = 0.25;
for i = 1:info.np
    pop(i, 1:data.n) = round(pop(i, 1:data.n) * (data.m - 1) + 1);
    fit(i) = decode(pop(i,:), info, data);
end
if info.num>0
    [info, sol, ~] = heft_sol_ccs(info, data);
    info.fitbest = decode(sol, info, data);
    pop(1, :) = sol;
    fit(1) = info.fitbest;
end
fitb = zeros(1, info.ng);
fithis = zeros(info.ng, info.np);
[f_best, best_idx] = min(fit);
global_best = pop(best_idx, :);
tic
for iter = 1:info.ng
    time_cost = toc;
    if time_cost > info.maxrt
        break;
    end
    fithis(iter, :) = fit;
    fitb(iter) = min(fit);
    fit_curve(iter) = min(fit);
    f_avg = mean(fit);
    Ex = f_best;
    En = f_avg - f_best;
    He = En / 5;
    sigma = normrnd(En, He);
    for i = 1:info.np
        mu_i = exp(-((fit(i) - Ex)^2) / (2 * sigma^2));
        alpha0 = info.gamma / (1 + exp(-mu_i));
        alpha_vec = alpha0 .* (pop(i,:) - global_best);
        levy_step = levy_flight(1.5, data.n * 3);
        new_solution = pop(i,:) + alpha_vec + levy_step;
        new_solution = min(max(new_solution, 0), 1);
        new_solution(1:data.n) = round(new_solution(1:data.n) * (data.m - 1) + 1);
        new_fit = decode(new_solution, info, data);
        if new_fit < fit(i)
            pop(i, :) = new_solution;
            fit(i) = new_fit;
        end
    end
    pa = info.pa * exp(-2 * iter / info.ng);
    for i = 1:info.np
        if rand < pa
            if rand < 0.5
                alpha_vec = alpha0 .* (pop(i,:) - global_best);
                levy_step = levy_flight(1.5, data.n * 3);
                new_solution = pop(i,:) + alpha_vec + levy_step;
            else
                new_solution = pop(i,:) + randn(1, data.n * 3) * 0.1;
            end
            new_solution = min(max(new_solution, 0), 1);
            new_solution(1:data.n) = round(new_solution(1:data.n) * (data.m - 1) + 1);
            new_fit = decode(new_solution, info, data);
            if new_fit < fit(i)
                pop(i,:) = new_solution;
                fit(i) = new_fit;
            end
        end
    end
    [curr_best, idx] = min(fit);
    if curr_best < f_best
        global_best = pop(idx, :);
        f_best = curr_best;
    end
    [worst_fit, worst_idx] = max(fit);
    if f_best < worst_fit
        pop(worst_idx, :) = global_best;
        fit(worst_idx) = f_best;
    end
end
energy = f_best;
time = toc;
end
function step = levy_flight(beta, dim)
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) / ...
              (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, dim) * sigma_u;
    v = randn(1, dim);
    step = u ./ abs(v).^(1 / beta);
end
function [info,sol,sch] = heft_sol_ccs(info,data)
dataheft = data;
info.m = dataheft.m;
info.n = size(dataheft.ct,1);
info.cmax = data.cmax;
dataheft.aft = dataheft.hx;
dataheft.w = dataheft.ct;
dataheft.ckef = dataheft.c;
dataheft.c = dataheft.st;
dataheft.pre = dataheft.xh;
dataheft.pkind = dataheft.p;
dataheft.ps = dataheft.pkind;
info.rank=rankqueen1(dataheft,info);
sch=heft(dataheft,info);
sol=initial_sol(sch,data);
sol = loc_energy(sol,data,info);
end
