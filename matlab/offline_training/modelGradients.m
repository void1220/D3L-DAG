% Legacy MATLAB scheduling routine.
function [gradients, loss] = modelGradients(parameters,dlX,sequenceLengths)
dlZ = modelEncoder(parameters,dlX,sequenceLengths);
state = struct;
state.HiddenState = dlZ;
state.CellState = zeros(size(dlZ),'like',dlZ);
dlY = modelDecoder(parameters,dlX,state);
dlYPred = dlY(:,:,1:end-1);
dlT = dlX(:,:,2:end);
loss = mean(maskedCrossEntropy(dlYPred,dlT,sequenceLengths));
gradients = dlgradient(loss,parameters);
sequenceLength = size(dlX,3);
loss = loss / sequenceLength;
end
