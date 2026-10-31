% Legacy MATLAB scheduling routine.
function [pX, pFit, fMin, bestX] = SSA(pop, info, data ,fit )
    c = 1;
    d = 0;
    dim = data.n*3;
    P_percent = 0.3;
    [pop_size, ~]=size(pop);
    pNum = round( pop_size *  P_percent );
    lb= c.*ones( 1,dim );
    ub= d.*ones( 1,dim );
    pFit = fit;
    pX = pop;
    [ fMin, bestI ] = min( fit );
    bestX = pop( bestI, : );
    M = 1000;
    for t = 1 : 1
        [ ~, sortIndex ] = sort( pFit );
        [fmax,B]=max( pFit );
        worse= pop(B,:);
        r2=rand(1);
        if(r2<0.9)
            for i = 1 : pNum
                r1=rand(1);
                pop( sortIndex( i ), : ) = pX( sortIndex( i ), : )*exp(-(i)/(r1* M));
                pop( sortIndex( i ), : ) = Bounds( pop( sortIndex( i ), : ), lb, ub, bestX);
                fit( sortIndex( i ) ) = decode_01( pop( sortIndex( i ), : ), info, data );
            end
        else
            for i = 1 : pNum
                pop( sortIndex( i ), : ) = pX( sortIndex( i ), : ) + randn(1) * ones(1,dim);
                pop( sortIndex( i ), : ) = Bounds( pop( sortIndex( i ), : ), lb, ub, bestX );
                fit( sortIndex( i ) ) = decode_01( pop( sortIndex( i ), : ),info,data );
            end
        end
        [ ~, bestII ] = min( fit );
        bestXX = pop( bestII, : );
        for i = ( pNum + 1 ) : pop_size
             A = floor(rand(1,dim)*2)*2-1;
            if( i>(pop_size/2))
               pop( sortIndex(i ), : )=randn(1)*exp((worse- pX( sortIndex( i ), : )) / (i)^2);
            else
                pop( sortIndex( i ), : )=bestXX + (abs(( pX( sortIndex( i ), : )-bestXX)))*(A'*(A*A')^(-1))*ones(1,dim);
            end
            pop( sortIndex( i ), : ) = Bounds( pop( sortIndex( i ), : ), lb, ub, bestX );
            fit( sortIndex( i ) ) = decode_01( pop( sortIndex( i ), : ),info,data );
        end
        c=randperm(numel(sortIndex));
        b=sortIndex(c(1:20));
        for j =  1  : length(b)
            if( pFit( sortIndex( b(j) ) ) > (fMin) )
                pop( sortIndex( b(j) ), : )=bestX+(randn(1,dim)).*(abs(( pX( sortIndex( b(j) ), : ) - bestX)));
            else
            pop( sortIndex( b(j) ), : ) =pX( sortIndex( b(j) ), : )+(2*rand(1)-1)*(abs(pX( sortIndex( b(j) ), : )-worse))/ ( pFit( sortIndex( b(j) ) )-fmax+1e-50);
            end
            pop( sortIndex(b(j) ), : ) = Bounds( pop( sortIndex(b(j) ), : ), lb, ub ,bestX);
            fit( sortIndex( b(j) ) ) = decode_01( pop( sortIndex( b(j) ), : ),info,data );
        end
        for i = 1 : pop_size
            if ( fit( i ) < pFit( i ) )
                pFit( i ) = fit( i );
                pX( i, : ) = pop( i, : );
            end
            if( pFit( i ) < fMin )
               fMin= pFit( i );
               bestX = pX( i, : );
            end
        end
    end
end
