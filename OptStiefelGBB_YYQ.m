function [W] = OptStiefelGBB_YYQ(W, daofj, derivative_of_rt, sumln, p,alpha, lambda,v)

[d,k] = size(W);
sumw = 0;
for i =1:k
    for j =1:d
      lnw(j) = log(1+abs(W(j,i)));
    end
    sumlnm = sum(lnw);
    sumw = sumw + sumlnm^2;
end





function [F, G] = fun( W, daofj, derivative_of_rt, sumln, p,alpha, lambda,v)
  G = daofj + derivative_of_rt; 
  F = alpha(v)^p + sumln + lambda*sumw; 
end


opts.record = 0; %
opts.mBitr  = 1000;
opts.Btol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;


tic; [W, out]= OptStiefelGBB(W, @fun, opts, daofj, derivative_of_rt, sumln, p,alpha, lambda,v); tsolve = toc;
out.fval = -2*out.fval;
end
