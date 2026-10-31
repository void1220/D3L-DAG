% Legacy MATLAB scheduling routine.
function [fit,able,fadv,pop_true]=decode(popin,info,data)
fp=popin(1:data.n);
info.n = data.n;
info.m = data.m;
info.cmax = data.cmax;
seq=zeros(1,info.n);
xh=data.xh;
hx=data.hx;
j=1;
[~, index]=sort(popin(info.n+1:2*info.n));
indexflag=zeros(1,info.n);
while j<info.n+1
    for i=1:info.n
        if all(xh(index(i),:)==0)&&(indexflag(i)==0)
            seq(j)=index(i);
            j=j+1;
            indexflag(i)=1;
            for k=1:size(hx,2)
                for l=1:size(xh,2)
                    if hx(index(i),k)~=0
                        if xh(hx(index(i),k),l)==index(i)
                            xh(hx(index(i),k),l)=0;
                            break
                        end
                    end
                end
            end
        end
    end
end
f=zeros(1,info.n);
for i=1:info.n
    f(i)=popin(info.n*2+i) * (1-data.f(fp(i)))   + data.f(fp(i));
end
xh=data.xh;
for k=1:data.m
    mt=zeros(1,info.m);
    st=zeros(1,info.n);
    dt=zeros(1,info.n);
    dn=zeros(1,info.n);
    for i=1:info.n
        curr=seq(i);
        dn(curr)=fp(curr);
        temp=find(xh(curr,:)>0);
        if ~isempty(temp)
            tst=zeros(1,length(temp));
            for j=1:length(temp)
                if dn(curr)~=dn(xh(curr,j))
                    tst(j)=dt(xh(curr,j))+data.st(xh(curr,j),curr);
                else
                    tst(j)=dt(xh(curr,j));
                end
            end
        end
        if ~isempty(temp)
            tlast=max(tst);
        else
            tlast=0;
        end
        st(curr)=max(mt(fp(curr)),tlast);
        dt(curr)=st(curr) + data.ct(curr,dn(curr)) / f(curr);
        mt(fp(curr))=dt(curr);
    end
end
if max(mt) > info.cmax
    able = 0;
else
    able = 1;
end
em=max(mt)*data.pks;
e=sum(em);
for i=1:info.n
    e = e + (   (  data.p(dn(i))   +   data.c(dn(i)) * f(i)^3 ) *  (   dt(i)-st(i)   )   );
end
fadv=zeros(1,info.n);
for i=1:info.n
    if popin(info.n*2+i)>=0.99
        fadv(i)=1;
    end
end
if max(mt)-info.cmax > 0
    fit = e + ( max(mt) - info.cmax ) * data.m/2 + data.n*2;
else
    fit = e;
end
pop_true = [popin(1:data.n), seq, f];
end
