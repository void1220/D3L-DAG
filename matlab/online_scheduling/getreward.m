% Legacy MATLAB scheduling routine.
function r=getreward(fitnew,fit,info)
n=0;
for i=1:info.np
   if fitnew(i)<fit(i)
      n=n+1;
   end
end
r=-0.5*abs(n/info.np-0.2);
