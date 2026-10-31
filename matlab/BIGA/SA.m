% Legacy MATLAB scheduling routine.
function [x0,f0]=SA(pop,data,info, Mmax)
    [pop_size, dim] = size(pop);
    lb = 0;
    ub = 1;
    l=lb.*ones(1,dim);
    u=ub.*ones(1,dim);
    info.n=data.n;
    info.m=data.m;
    info.cmax=data.cmax;
    Best_x_f=inf;
    Best_x=zeros(1,dim);
    x0 = pop;
    f0 = decode_01( x0, info, data );
    for i = 1 : pop_size
        if f0(i) < Best_x_f
            Best_x_f = f0(i);
            Best_x = x0(i,:);
        end
    end
    for m = 1 : 1
        T=m/Mmax;
        mu=10^(T*1000);
        for k = 1:1
            dx=mu_inv(2*rand(1,dim)-1,mu).*(u-l);
            if isnan(dx)
                break
            end
            x1=x0+dx;
            for i = 1:pop_size
                x1(i,:) = Bounds( x1(i,:), u, l, Best_x);
            end
            fx1 = decode_01( x1, info, data );
            for i = 1 : pop_size
                if fx1(i) < f0(i)
                    x0(i,:) = x1(i,:);
                    f0(i) = fx1(i);
                    if fx1(i) < Best_x_f
                        Best_x_f = fx1(i);
                        Best_x = x0(i,:);
                    end
                end
            end
        end
    end
end
function x=mu_inv(y,mu)
    x=(((1+mu).^abs(y)-1)/mu).*sign(y);
end
