% Legacy MATLAB scheduling routine.
function sch=heft(data,info)
sch.xij=zeros(1,info.n);
sch.mkTtemp = zeros(1,info.m);
sch.st = zeros(1, info.n);
sch.et = zeros(1,info.n);
sch.aft= zeros(1,info.n);
for i=1:info.n
    taskID=info.rank(i);
    temp= data.pre(taskID,:)>0;
    curpre=data.pre(taskID,temp);
   if ~isempty(curpre)
        eft=zeros(1,info.m);
       for k=1:info.m
           abst=zeros(1,length(curpre));
          for j=1:length(curpre)
                if sch.xij(curpre(j))==k
                   abst(j)=sch.st(curpre(j))+sch.et(curpre(j));
                else
                    abst(j)=sch.st(curpre(j))+sch.et(curpre(j))+data.c(curpre(j),taskID);
                end
          end
          [abstmax,~]=max(abst);
           eft(k)=max(abstmax+data.w(taskID,k),sch.mkTtemp(k)+data.w(taskID,k));
       end
        [~,index1]=min(eft);
        sch.xij(taskID)=index1;
        sch.st(taskID)=min(eft)-data.w(taskID,sch.xij(taskID));
        sch.et(taskID)=data.w(taskID,sch.xij(taskID));
        sch.mkTtemp(sch.xij(taskID))=sch.st(taskID)+sch.et(taskID);
    else
        sch.st(taskID)=0;
        [~,index]=min(data.w(taskID,:));
        sch.xij(taskID)=index;
        sch.et(taskID)=data.w(taskID,sch.xij(taskID));
        sch.mkTtemp(sch.xij(taskID))=sch.st(taskID)+sch.et(taskID);
   end
   sch.aft(taskID)= sch.st(taskID)+ sch.et(taskID);
end
sch.LB=sch.st(info.n)+sch.et(info.n);
E=zeros(1,info.n);
Etolal=0;
for i=1:info.n
    E(i)=(data.pkind(sch.xij(i))+data.ckef(sch.xij(i)))*data.w(i,sch.xij(i));
    Etolal=Etolal+ E(i);
end
ES=sum(info.m*data.ps*sch.LB);
sch.Etolal=Etolal+ES;
end
