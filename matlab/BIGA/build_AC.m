% Legacy MATLAB scheduling routine.
function [actor,critic,agent] = build_AC(obsInfo, actInfo)
    input_layer = featureInputLayer(prod(obsInfo.Dimension), 'Name', 'input');
    fc1 = fullyConnectedLayer(64, 'Name', 'fc1');
    relu1 = reluLayer('Name', 'relu1');
    zheng = dropoutLayer(0.5);
    fc2 = fullyConnectedLayer(32, 'Name', 'fc2');
    relu2 = reluLayer('Name', 'relu2');
    output_layer = fullyConnectedLayer(numel(actInfo.Elements), 'Name', 'output');
    softmax_layer = softmaxLayer('Name', 'softmax');
    net = [
        input_layer
        fc1
        relu1
        zheng
        fc2
        relu2
        output_layer
        softmax_layer
        ];
    net = dlnetwork(net);
    actor = rlDiscreteCategoricalActor(net,obsInfo,actInfo);
    input_layer = featureInputLayer(prod(obsInfo.Dimension), 'Name', 'input');
    fc1 = fullyConnectedLayer(64, 'Name', 'fc1');
    relu1 = reluLayer('Name', 'relu1');
    fc2 = fullyConnectedLayer(32, 'Name', 'fc2');
    relu2 = reluLayer('Name', 'relu2');
    output_layer = fullyConnectedLayer(1, 'Name', 'output');
    layers = [
        input_layer
        fc1
        relu1
        fc2
        relu2
        output_layer
    ];
    dlnet = dlnetwork(layers);
    critic = rlValueFunction(dlnet,obsInfo);
    actor.UseDevice = 'gpu';
    critic.UseDevice = 'gpu';
    actorOpts = rlOptimizerOptions('LearnRate',0.005,'GradientThreshold',1);
    criticOpts = rlOptimizerOptions('LearnRate',1e-3,'GradientThreshold',1);
    agentOptions = rlACAgentOptions(...
        'NumStepsToLookAhead', 16, ...
        'DiscountFactor', 0.95, ...
        'ActorOptimizerOptions', actorOpts, ...
        'CriticOptimizerOptions', criticOpts, ...
        'SampleTime', 1);
    agent = rlACAgent(actor, critic, agentOptions);
    agent.UseExplorationPolicy = true;
end
