% Legacy MATLAB scheduling routine.
function popout=cr(pop,fit,pc)
[ps,l]=size(pop);
popout=pop;
n=1;
sumnum=0;
for i=1:ps
    sumnum=sumnum+(1/fit(i));
end
p=zeros(1,ps);
for i=1:ps
    p(i)=(1/fit(i)).^n/sumnum;
end
p=[0 p];
for i=1:ps/2
    temp=rand(1,1);
    if temp<pc
        temp=rand(1,1);
        for j=1:ps
            if (temp>=sum(p(1:j)))&&(temp<sum(p(1:j+1)))
               index(1)=j;
            end
        end
        temp=rand(1,1);
        for j=1:ps
            if (temp>=sum(p(1:j)))&&(temp<sum(p(1:j+1)))
               index(2)=j;
            end
        end
        while index(1)==index(2)
            temp=rand(1,1);
            for j=1:ps
                if (temp>=sum(p(1:j)))&&(temp<sum(p(1:j+1)))
                    index(2)=j;
                end
            end
        end
        tempc=randperm(l);
        b=tempc(1);
        c=tempc(2);
        if b<c
           temp=b;
           b=c;
           c=temp;
        end
        popout(i*2-1,:)=pop(index(1),:);
        popout(i*2,:)=pop(index(2),:);
        popout(i*2-1,b:c)=pop(index(2),b:c);
        popout(i*2,b:c)=pop(index(1),b:c);
    end
end
