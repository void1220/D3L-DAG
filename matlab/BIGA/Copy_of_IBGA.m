% Legacy MATLAB scheduling routine.
function [energy, time, fit_curve] = IBGA(info, data)
    info.n = data.n;
    info.m = data.m;
    info.cmax = data.cmax;
    info.pc = 0.6;
    info.pm = 0.5;
    info.er = 0.2;
    pop = initialize_IBGA(info, data);
    [fit, able, ~] = decode_01(pop, info, data);
    fit_curve = zeros(1, info.ng);
    tic;
    for gen = 1:info.ng
        time_cost = toc;
        if time_cost > info.maxrt
            break
        end
        pop = elitism_selection_IBGA(pop, fit, info);
        pop = crossover_IBGA(pop, info);
        pop = mutation_IBGA(pop, info);
        [fit, able, ~] = decode_01(pop, info, data);
        fit_curve(gen) = min(fit);
    end
    energy = min(fit);
    time = toc;
end
function pop = initialize_IBGA(info, data)
    n = data.n;
    popsize = info.np;
    pop = zeros(popsize, n * 3);
    for i = 1:popsize
        pop(i,1:n) = 0.5 + 0.1 * randn(1,n);
        pop(i,n+1:n*2) = rand(1,n);
        pop(i,n*2+1:n*3) = rand(1,n);
    end
end
function pop_new = elitism_selection_IBGA(pop, fit, info)
    [~, idx] = sort(fit);
    elite_num = round(info.np * info.er);
    elite = pop(idx(1:elite_num), :);
    f = 1 ./ fit;
    f = f / sum(f);
    f = cumsum(f);
    pop_new = elite;
    for i = 1:(info.np - elite_num)
        sel = find(rand <= f, 1, 'first');
        pop_new = [pop_new; pop(sel, :)];
    end
end
function pop = crossover_IBGA(pop, info)
    [ps, len] = size(pop);
    n = info.n;
    ncr = round(ps * info.pc);
    for i = 1:2:ncr
        i1 = randi(ps);
        i2 = randi(ps);
        pt1 = randi(n);
        pop([i1,i2],1:n) = ...
            [pop(i1,1:pt1), pop(i2,pt1+1:n); pop(i2,1:pt1), pop(i1,pt1+1:n)];
        m1 = rand(1,n) > 0.5;
        tmp1 = pop(i1,n+1:n*2);
        tmp2 = pop(i2,n+1:n*2);
        pop(i1,n+1:n*2) = m1 .* tmp1 + ~m1 .* tmp2;
        pop(i2,n+1:n*2) = ~m1 .* tmp1 + m1 .* tmp2;
        mask = rand(1,n) > 0.5;
        pop(i1,n*2+1:end) = mask .* pop(i1,n*2+1:end) + ~mask .* pop(i2,n*2+1:end);
        pop(i2,n*2+1:end) = ~mask .* pop(i1,n*2+1:end) + mask .* pop(i2,n*2+1:end);
    end
end
function pop = mutation_IBGA(pop, info)
    [ps, ~] = size(pop);
    n = info.n;
    nmu = round(ps * info.pm);
    for i = 1:nmu
        idx = randi(ps);
        j = randperm(n,2);
        pop(idx,j(1)) = rand();
        pop(idx,j(2)) = rand();
        ord = pop(idx,n+1:n*2);
        p1 = randi(n); p2 = randi(n);
        val = ord(p1);
        ord(p1) = [];
        ord = [ord(1:p2-1), val, ord(p2:end)];
        pop(idx,n+1:n*2) = ord;
        j = randi(n);
        pop(idx,n*2 + j) = rand();
    end
end
function pop = VNS_IBGA(pop, info)
    [ps, ~] = size(pop);
    n = info.n;
    for i = 1:round(ps * 0.1)
        idx = randi(ps);
        ord = pop(idx,n+1:n*2);
        pos = randperm(n,2);
        ord([pos(1),pos(2)]) = ord([pos(2),pos(1)]);
        pop(idx,n+1:n*2) = ord;
    end
end
