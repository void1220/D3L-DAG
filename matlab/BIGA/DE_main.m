% Legacy MATLAB scheduling routine.
function [info,pop] = DE_main(pop,data,info,fit)
    info.n=data.n;
    info.m=data.m;
    info.h=1;
    info.cmax=data.cmax;
    pop_size = size(pop, 1);
    dim = size(pop, 2);
    F = 0.001;
    max_gen = 1;
    gen = 1;
    cr = 0.5;
    v = zeros(pop_size, dim);
    v1 = zeros(pop_size, dim);
    v2 = zeros(pop_size, dim);
    while gen <= max_gen
        for i = 1:pop_size
            r1 = randi(pop_size);
            while r1 == i
                r1 = randi(pop_size);
            end
            r2 = randi(pop_size);
            while r2 == i || r2 == r1
                r2 = randi(pop_size);
            end
            r3 = randi(pop_size);
            while r3 == i || r3 == r1 || r3 == r2
                r3 = randi(pop_size);
            end
            r4 = randi(pop_size);
            while r4 == i || r4 == r1 || r4 == r2 || r4 == r3
                r4 = randi(pop_size);
            end
            [~,best]=min(fit);
            old_pop = pop;
            [sorted,index] = sort(fit);
            min_index = index(1:3);
            best1_pop = pop(min_index(1),:);
            best2_pop = pop(min_index(2),:);
            best3_pop = pop(min_index(3),:);
            v_i = F * (pop(r1,:) - pop(r2,:))  + F * (pop(r3,:) - pop(r4,:));
            u = pop(r4,:) + v_i;
            for j = 1: data.n*3
                if rand() < cr
                    pop(i, j)=u(j);
                end
            end
            pop_i = pop(i,:);
            j_rand = randi([1 data.n*3]);
            for j = 1:data.n*3
                if rand() < cr || j == j_rand
                    pop_i(j) = u(j);
                end
            end
            pop(i,:) = pop_i;
            for j = 1:data.n*3
                if pop(i,j) <= 0 || pop(i,j) >= 1  && rand() < 0.8
                    pop(i,j) = old_pop(i,j);
                end
            end
        end
        gen = gen+1;
    end
end
