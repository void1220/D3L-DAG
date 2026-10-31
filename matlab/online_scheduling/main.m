% Legacy MATLAB scheduling routine.
clear
clc
load datain.mat
res_e=zeros(32,7);
res_t=zeros(32,7);
for M=1:7
    for K=1:2
        for I=1:16
            info.mode = K;
            info.num  = I;
            info.flag = M;
            if info.mode==1
                temp=['data=datain.fft',num2str(info.num),';'];
            elseif info.mode==2
                temp=['data=datain.gs',num2str(info.num),';'];
            else
                error('unavailable');
            end
            eval(temp)
            info=setuppara(info);
            if info.flag==1
                res_e((info.mode-1)*16+info.num,1)=EHEFT(info,data);
            elseif info.flag==2
                [res_e((info.mode-1)*16+info.num,2),res_t((info.mode-1)*16+info.num,2)]=IILS(info,data);
            elseif info.flag==3
                [res_e((info.mode-1)*16+info.num,3),res_t((info.mode-1)*16+info.num,3)]=ILS(info,data);
            elseif info.flag==4
                [res_e((info.mode-1)*16+info.num,4),res_t((info.mode-1)*16+info.num,4)]=GA(info,data);
            elseif info.flag==5
                [res_e((info.mode-1)*16+info.num,5),res_t((info.mode-1)*16+info.num,5)]=ACS(info,data);
            elseif info.flag==6
                [res_e((info.mode-1)*16+info.num,6),res_t((info.mode-1)*16+info.num,6)]=LGWO(info,data);
            elseif info.flag==7
                [res_e((info.mode-1)*16+info.num,6),res_t((info.mode-1)*16+info.num,6)]=AILS(info,data);
            else
                error('unavailable');
            end
        end
    end
end
