% Legacy MATLAB scheduling routine.
function reward = RL_reward(fit, fitold, info)
omega=0.2;
alpha=0.7;
epsilon = 1e-8;
reward_local = (fitold - fit) ./ (fitold + epsilon);
reward_local = max(min(reward_local, 1), -1);
reward_improvement = 0.9 * sum(reward_local) + 0.1 * mean(reward_local);
beneficial = double(fit < fitold);
beneficial_ratio = sum(beneficial) / info.np;
reward_ratio_error = abs(beneficial_ratio - omega);
reward_ratio_score = 1 - reward_ratio_error;
reward = alpha * reward_improvement + (1 - alpha) * reward_ratio_score;
end
