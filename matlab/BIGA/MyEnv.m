% Legacy MATLAB scheduling routine.
classdef MyEnv< rl.env.MATLABEnvironment
    properties
        data = []
        info = []
        PopulationSize = 40
        IndividualLength
        NumOperators = 20
        StateSize = 4
        ActionSize
        CurrentPopulation
        CurrentState
        fitnew
        fitold
    end
    methods
        function this = MyEnv(data, info)
            ObservationInfo = rlNumericSpec([1 4]);
            ObservationInfo.Name = 'Population State';
            ObservationInfo.Description = 'Standard Deviation and Mean of Fitness';
            ActionInfo = rlFiniteSetSpec({1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20});
            ActionInfo.Name = 'Operator Action';
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            this.data = data;
            this.info = info;
            this.IndividualLength = data.n * 3;
            this.ActionSize = this.NumOperators;
            this.CurrentPopulation = RL_creat_pop(data, info);
            this.fitold = decode_01(this.CurrentPopulation,info,data);
            this.fitnew = this.fitold;
            [inputFeatures, ~] = calculate_input(this.CurrentPopulation, this.info, this.data, this.fitnew, this.fitold);
            this.CurrentState = round(inputFeatures);
        end
        function [Observation, Reward, IsDone, Info] = step(this, Action)
            [~, ~, oldfit] = calculate_std(this.CurrentPopulation, this.info, this.data);
            [this.info, newPopulation,newfit] = performOperator(this.CurrentPopulation, Action, this.info, this.data, oldfit);
            this.fitnew = newfit;
            [inputFeatures, ~] = calculate_input(newPopulation, this.info, this.data, newfit, oldfit);
            newState = inputFeatures;
            Reward = RL_reward(newfit, oldfit, this.info);
            this.CurrentPopulation = newPopulation;
            this.CurrentState = newState;
            Observation = this.CurrentState;
            IsDone = false;
            Info = [];
        end
        function InitialObservation = reset(this)
            this.CurrentPopulation = RL_creat_pop(this.data, this.info);
            this.fitold = decode_01(this.CurrentPopulation,this.info,this.data);
            this.fitnew = this.fitold;
            [inputFeatures, ~] = calculate_input(this.CurrentPopulation, this.info, this.data, this.fitnew, this.fitold);
            InitialObservation = this.CurrentState;
        end
    end
end
function pop = RL_creat_pop(data,info)
pop=rand(info.np,data.n*3);
end
function [fit,able,pop_true] = decode_01(pop,info,data)
pop_size = size(pop,1);
fit = zeros(1,pop_size);
pop_true = zeros(info.np, data.n*3);
able = zeros(1,pop_size);
for i = 1 : pop_size
    pop(i,1:data.n) = ceil(pop(i,1:data.n) * data.m);
    [~,pop(i,data.n+1:data.n*2)] = sort(pop(i,data.n+1:data.n*2) * data.n);
end
for k = 1 : pop_size
    [fit(k),able(k),~,pop_true(k,:)] = decode(pop(k,:),info,data);
end
end
function [inputFeatures, fit] = calculate_input(pop,info,data,fit,fitold)
if info.mode == 1
    min_fit = [181.51 195.26 220.51 269.26 300 325.92 372.45 402.79 767.32 746.49 2000 2000 2000 2000 2200 2200];
elseif info.mode == 2
    min_fit = [198.48 142.96 343.69 266.47 382.24 274.59 791.44 686.03 919.15 743.23 2000 1700 2400 3000 3000 3000];
end
standardDeviation = std(fit) / 1000;
mean_fit = min_fit(info.num) / mean(fit);
min_fit = min_fit(info.num) / min(fit);
sum_ev = 0;
for i = 1:info.np
    sum_ev = sum_ev + ( fitold(i) - fit(i) );
end
E_V = sqrt( info.np * ( min(fitold) - min(fit) ) ) / ( sum_ev + 1 )  ;
inputFeatures = [standardDeviation, mean_fit,min_fit,E_V];
end
function [standardDeviation, mean_fit,fit] = calculate_std(pop,info,data)
fit = decode_01(pop,info,data);
stdfit = std(fit);
standardDeviation = stdfit;
mean_fit = mean(fit);
end
function [info,newPopulation,fit] = performOperator(pop, action,info,data,fit)
[info,newPopulation,fit]=RL_action(pop,data,info,action,fit);
end
function [info,pop,fit]=RL_action(pop,data,info,action,fit)
if action == 1
    [info,pop,fit]=operator_1(pop,data,info,fit);
elseif action == 2
    [info,pop,fit]=operator_2(pop,data,info,fit);
elseif action == 3
    [info,pop,fit]=operator_3(pop,data,info,fit);
elseif action == 4
    [info,pop,fit]=operator_4(pop,data,info,fit);
elseif action == 5
    [info,pop,fit]=operator_5(pop,data,info,fit);
elseif action == 6
    [info,pop,fit]=operator_6(pop,data,info,fit);
elseif action == 7
    [info,pop,fit]=operator_7(pop,data,info,fit);
elseif action == 8
    [info,pop,fit]=operator_8(pop,data,info,fit);
elseif action == 9
    [info,pop,fit]=operator_9(pop,data,info,fit);
elseif action == 10
    [info,pop,fit]=operator_10(pop,data,info,fit);
elseif action == 11
    [info,pop,fit]=operator_11(pop,data,info,fit);
elseif action == 12
    [info,pop,fit]=operator_12(pop,data,info,fit);
elseif action == 13
    [info,pop,fit]=operator_13(pop,data,info,fit);
elseif action == 14
    [info,pop,fit]=operator_14(pop,data,info,fit);
elseif action == 15
    [info,pop,fit]=operator_15(pop,data,info,fit);
elseif action == 16
    [info,pop,fit]=operator_16(pop,data,info,fit);
elseif action == 17
    [info,pop,fit]=operator_17(pop,data,info,fit);
elseif action == 18
    [info,pop,fit]=operator_18(pop,data,info,fit);
elseif action == 19
    [info,pop,fit]=operator_19(pop,data,info,fit);
elseif action == 20
    [info,pop,fit]=operator_20(pop,data,info,fit);
end
end
