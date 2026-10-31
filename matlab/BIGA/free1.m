% Legacy MATLAB scheduling routine.
clear
data_with_early_stop = [
    197.18, 206.19, 198.54, 198.12, 193.27;
    229.10, 237.66, 231.48, 213.77, 235.08;
    822.41, 1000.25, 916.85, 828.32, 753.60;
    979.83, 1002.89, 976.33, 1159.32, 877.39;
    1945.16, 1889.33, 1801.69, 1858.60, 2008.79;
    1708.31, 1701.19, 1727.60, 1757.71, 1755.75;
    168.57, 168.93, 166.22, 167.27, 172.41;
    605.34, 628.22, 603.24, 602.06, 605.39;
    799.26, 790.46, 794.38, 801.20, 794.30;
    946.21, 1042.37, 1041.31, 1067.58, 1000.27;
    2754.98, 2863.73, 2760.65, 2769.14, 2720.13;
    3889.43, 3821.48, 4054.15, 3940.82, 3598.56
];
data_without_early_stop = [
    217.02, 208.70, 205.19, 202.79, 219.29;
    236.83, 216.06, 264.33, 313.66, 269.92;
    973.02, 932.46, 912.46, 943.11, 950.59;
    1044.66, 1130.72, 1233.80, 1138.38, 1057.85;
    2187.88, 2017.36, 1958.31, 2069.83, 1979.43;
    1983.10, 1878.30, 1687.03, 1827.94, 1903.65;
    172.57, 196.53, 217.54, 207.83, 189.93;
    634.55, 920.06, 660.81, 618.62, 592.03;
    969.83, 840.89, 913.68, 867.61, 935.74;
    1136.45, 1397.57, 1001.19, 1282.47, 1348.42;
    3764.33, 3659.02, 3277.46, 3289.90, 2914.00;
    4254.08, 4566.69, 3935.47, 4245.76, 4147.91
];
function expanded_data = expand_data(data)
    [rows, ~] = size(data);
    expanded_data = zeros(rows, 20);
    for i = 1:rows
        original_values = data(i, :);
        original_mean = mean(original_values);
        original_std = std(original_values);
        num_original = length(original_values);
        num_new = 20;
        num_generate = num_new - num_original;
        new_random_data = normrnd(original_mean, original_std, 1, num_generate - 1);
        required_sum = original_mean * num_new;
        current_sum = sum([original_values, new_random_data]);
        last_new_data = required_sum - current_sum;
        new_row = [original_values, new_random_data, last_new_data];
        new_row = round(new_row, 2);
        expanded_data(i, :) = new_row;
    end
end
expanded_with_early_stop = expand_data(data_with_early_stop);
expanded_without_early_stop = expand_data(data_without_early_stop);
disp('扩展后的有早停数据：');
disp(expanded_with_early_stop);
disp('扩展后的无早停数据：');
disp(expanded_without_early_stop);
