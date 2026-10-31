% Legacy MATLAB scheduling routine.
function [info,solout]=insert_seq_LSTM(sol,data,info,seq_net)
test_data = [1 sol(data.n+1:2*data.n)+1 data.n+2];
fit=decode(sol,info,data);
solnew = sol;
parameters = seq_net;
sequenceLength=length(test_data);
dlX = dlarray(test_data,'BTC');
dlZ = modelEncoder(parameters,dlX,sequenceLength);
lstm_sol=double(dlZ);
soltemp=lstm_sol;
for i=1:length(lstm_sol)-1
    for j=i:length(lstm_sol)
        soltemp(j)=lstm_sol(i);
        soltemp(i:j-1)=lstm_sol(i+1:j);
        state = struct;
        state.HiddenState = dlZ;
        state.CellState = zeros(size(dlZ),'like',dlZ);
        dlY = modelDecoder(parameters,dlX,state);
        [~,idx] = max(dlY,[],1);
        idx = squeeze(idx)';
        res = idx(1:end-2);
        solnew(data.n+1:2*data.n)=res-1;
        for k=data.n+1:2*data.n
            if solnew(k)~=sol(k)
                temp=rand(1,1);
                if temp<0.9
                    solnew(k)=sol(k);
                end
            end
        end
        for k=data.n+1:2*data.n
            if (solnew(k)>data.n)||(solnew(k)<1)
                solnew(k)=sol(k);
            end
        end
        fittemp=decode(solnew,info,data);
        if fittemp<fit
            sol=solnew;
            if fittemp<info.fitbest
                info.solbest=sol;
                info.fitbest=fittemp;
            end
        else
            temp=rand(1,1);
            if exp((fit-fittemp)/fit)>temp
                sol=solnew;
            end
        end
    end
end
solout=info.solbest;
