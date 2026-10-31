% Legacy MATLAB scheduling routine.
function solnew=loc_energy(sol,data,info)
soltemp=sol;
info.n = data.n;
info.m = data.m;
info.cmax = data.cmax;
for i=1:1000
    soltemp(data.n*2+1:end)=sol(data.n*2+1:end)*(1-i/1000);
    ab=jugg(soltemp,info,data);
    if ab==0
        break
    end
end
N=i-1;
soltemp(data.n*2+1:end)=normrnd((i-1)/1000,0.05,1,data.n);
for i=1:data.n
    if soltemp(data.n*2+i)<=0
        soltemp(data.n*2+i)=rand(1,1);
    elseif soltemp(data.n*2+i)>=1
        soltemp(data.n*2+i)=rand(1,1);
    end
end
index=randperm(data.n);
for i=1:data.n
    soltemp(data.n*2+index(i))=unifrnd(1-N/1000,1);
    ab=jugg(soltemp,info,data);
    if ab==1
        break
    end
end
solnew=soltemp;
end
