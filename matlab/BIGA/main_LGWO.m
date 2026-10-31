% Legacy MATLAB scheduling routine.
clear; clc;
load datain_rl.mat;
algorithm_list = {'GA', 'SACS', 'LGWO'};
iteraction = 5;
res_e = nan(10, 3, iteraction);
for mode = 1:2
    for num = 1:5
        for algorithm = 1:3
            info.flag = algorithm;
            info.mode = mode;
            info.num = num;
            if mode == 1
                eval(sprintf('data = datain_rl.fft%d;', num));
                mode_str = 'fft';
            else
                eval(sprintf('data = datain_rl.gs%d;', num));
                mode_str = 'gs';
            end
            info = setuppara(info);
            idx_row = (mode - 1) * 5 + num;
            idx_col = algorithm;
            for iter = 1:iteraction
                switch algorithm
                    case 1, [e_iter, ~] = GA(info, data);
                    case 2, [e_iter, ~] = SACS(info, data);
                    case 3, [e_iter, ~] = LGWO(info, data);
                end
                res_e(idx_row, idx_col, iter) = e_iter;
                fprintf('%s  \t%s%d \t Iter %d\t res_e: %.2f\n', ...
                    algorithm_list{idx_col}, mode_str, num, iter, e_iter);
            end
        end
    end
end
