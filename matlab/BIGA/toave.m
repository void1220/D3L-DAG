% Legacy MATLAB scheduling routine.
load('E:\code_AE\0_now_RL\3_main_test/RESULT/result0.mat');
[rows, cols, iterations] = size(res_e);
avg_res_e = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        avg_res_e(i, j) = mean(res_e(i, j, :));
    end
end
