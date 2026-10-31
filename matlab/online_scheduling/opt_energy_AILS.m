% Legacy MATLAB scheduling routine.
function sol=opt_energy_AILS(sol,data,info)
soltemp=sol;
for i=1:1000
    soltemp(data.n*2+1:end)=sol(data.n*2+1:end)*(1-i/1000);
    ab=jugg(soltemp,info,data);
    if ab==0
        break
    end
end
N=i-1;
sol(data.n*2+1:end)=sol(data.n*2+1:end)*(1-N/1000);
