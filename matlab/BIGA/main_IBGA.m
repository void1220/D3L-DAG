% Legacy MATLAB scheduling routine.
clear
clc
load datain_rl.mat
iteraction = 5;
res_e = zeros(32, 7, iteraction);
res_t = zeros(32, 7, iteraction);
avg_res_e = zeros(32, 7);
avg_res_t = zeros(32, 7);
max_min_res_e = zeros(32, 7);
max_min_res_t = zeros(32, 7);
algorithm_list = {'EHEFT', 'ILS', 'IILS', 'IBGA', 'LGWO', 'CCS', 'Trans_RL'};
for algorithm = 4
    for mode = 1
        for num = 16:-1:1
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
                if info.flag == 1
                    e_iter = EHEFT(info, data);
                    t_iter = 0;
                elseif info.flag == 2
                    [e_iter, t_iter] = ILS(info, data);
                elseif info.flag == 3
                    [e_iter, t_iter] = IILS(info, data);
                elseif info.flag == 4
                    [e_iter, t_iter] = IBGA(info, data);
                elseif info.flag == 5
                    [e_iter, t_iter] = LGWO(info, data);
                elseif info.flag == 6
                    [e_iter, t_iter] = CCS(info, data);
                elseif info.flag == 7
                    [e_iter, t_iter] = Trans_RL(info, data);
                else
                    error('unavailable');
                end
                res_e(idx_row, idx_col, iter) = e_iter;
                res_t(idx_row, idx_col, iter) = t_iter;
                fprintf('%s%d %s Iteration %d: res_e: %.4f, res_t: %.4f\n', mode_str, info.num, algorithm_list{idx_col}, iter, e_iter, t_iter);
            end
            avg_res_e(idx_row, idx_col) = mean(res_e(idx_row, idx_col, :));
            avg_res_t(idx_row, idx_col) = mean(res_t(idx_row, idx_col, :));
            save('res_e.mat', 'res_e','-append');
            save('res_t.mat', 'res_t','-append');
            save('avg_res_e.mat', 'avg_res_e','-append');
            save('avg_res_t.mat', 'avg_res_t','-append');
        end
    end
end
