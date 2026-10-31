% Legacy MATLAB scheduling routine.
clear
clc
load fft6train.mat
embeddingDimension = 80;
numHiddenUnits = 70;
ProblemSize = data.n + 2;
latentDimension = round(data.n*0.4);
seq_net = struct;
mu = 0;
sigma = 0.01;
seq_net.emb.Weights = initializeGaussian([embeddingDimension ProblemSize],mu,sigma);
sz = [4*numHiddenUnits embeddingDimension];
numOut = 4*numHiddenUnits;
numIn = embeddingDimension;
seq_net.lstmEncoder.InputWeights = initializeGlorot(sz,numOut,numIn);
seq_net.lstmEncoder.RecurrentWeights = initializeOrthogonal([4*numHiddenUnits numHiddenUnits]);
seq_net.lstmEncoder.Bias = initializeUnitForgetGate(numHiddenUnits);
sz = [latentDimension numHiddenUnits];
numOut = latentDimension;
numIn = numHiddenUnits;
seq_net.fcEncoder.Weights = initializeGlorot(sz,numOut,numIn);
seq_net.fcEncoder.Bias = initializeZeros([latentDimension 1]);
sz = [4*latentDimension embeddingDimension];
numOut = 4*latentDimension;
numIn = embeddingDimension;
seq_net.lstmDecoder.InputWeights = initializeGlorot(sz,numOut,numIn);
seq_net.lstmDecoder.RecurrentWeights = initializeOrthogonal([4*latentDimension latentDimension]);
seq_net.lstmDecoder.Bias = initializeZeros([4*latentDimension 1]);
sz = [ProblemSize latentDimension];
numOut = ProblemSize;
numIn = latentDimension;
seq_net.fcDecoder.Weights = initializeGlorot(sz,numOut,numIn);
seq_net.fcDecoder.Bias = initializeZeros([ProblemSize 1]);
miniBatchSize = 128;
numEpochs = 100;
learnRate = 0.01;
plots = "training-progress";
executionEnvironment = "auto";
trailingAvg = [];
trailingAvgSq = [];
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    xlabel("Iteration")
    ylabel("Loss")
    ylim([0 inf])
    grid on
end
numObservations = numel(seq_train);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);
iteration = 0;
start = tic;
for epoch = 1:numEpochs
    idx = randperm(numObservations);
    seq_train = seq_train(idx);
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        documentsBatch = seq_train(idx);
        X=cell(miniBatchSize,1);
        for j=1:miniBatchSize
            X{j} = [1 seq_train{j}+1 ProblemSize];
            if length(X{j}) < ProblemSize
                X{j} = [X{j} ones(1,ProblemSize-length(X{j}))*ProblemSize];
            end
        end
        X = cat(1,X{:});
        dlX = dlarray(X,'BTC');
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        sequenceLengths = zeros(miniBatchSize,1);
        for j=1:miniBatchSize
            sequenceLengths(j)=length(seq_train{j})+1;
        end
        [gradients,loss] = dlfeval(@modelGradients, seq_net, dlX, sequenceLengths);
        [seq_net,trailingAvg,trailingAvgSq] = adamupdate(seq_net,gradients, ...
            trailingAvg,trailingAvgSq,iteration,learnRate);
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end
save seq_fft6_net.mat
