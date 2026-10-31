% Legacy MATLAB scheduling routine.
function s = Bounds( s, Ub, Lb , best_s)
    temp = s;
    I = temp <= Lb;
    if rand() < 0.01
    temp(I) = best_s(I);
    else
    temp(I) = rand();
    end
    J = temp >= Ub;
    if rand() < 0.01
        temp(J) = best_s(J);
    else
        temp(J) = rand();
    end
      s = temp;
end
