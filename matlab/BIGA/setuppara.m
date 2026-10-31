% Legacy MATLAB scheduling routine.
function info = setuppara(info)
    if info.mode == 1
        if info.num <= 4
            info.maxrt = 10;
        elseif info.num <= 8
            info.maxrt = 30;
        elseif info.num <= 10
            info.maxrt = 150;
        elseif info.num <= 16
            info.maxrt = 300;
        end
    elseif info.mode == 2
        if info.num <= 2
            info.maxrt = 10;
        elseif info.num <= 6
            info.maxrt = 30;
        elseif info.num <= 10
            info.maxrt = 150;
        elseif info.num <= 16
            info.maxrt = 300;
        end
    end
    if info.mode == 1
        if info.num <= 3
            info.ng = 2000;
        elseif info.num <= 16
            info.ng = 3000;
        end
    else
        if info.num <= 2
            info.ng = 1500;
        elseif info.num <= 16
            info.ng = 10000;
        end
    end
    info.pc = 0.9;
    info.pm = 0.2;
    info.tf = 500;
    info.tg = 100;
    info.tl = 5;
    info.tl1 = 8;
    info.tl2 = 5;
    info.tl3 = 2;
    info.np = 40;
end
