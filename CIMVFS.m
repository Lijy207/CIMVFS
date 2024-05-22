function [ WW,W, alpha ] = CIMVFS(X1,Y,para )
%   CIMVFS
%   Efficient Multi-View Feature Selection for Classifying Imbalanced Data

Dataa=X1;
%% Leaves dataset
Data1 = Dataa(:,1:64);
Data2 = Dataa(:,65:128);
Data3 = Dataa(:,129:192);
Data5 = {Data1,Data2,Data3};

%% Initialization
c=size(unique(Y),1);
p = 2;
alpha = rand(3,1);
WW = [];
for v = 1:3
    X = cell2mat(Data5(v))';
    [d,n] = size(X);
    [m,~] = size(Y);
    k = 64-1;
    bias = rand(k,k);
    lambda = para.lambda;
    mu = [];
    for j =1:c
        idx = find(Y==j);
        mu(:,j) = mean(X(:,idx),2);
        Xidx = X(:,idx);
        Sigma{j} = cov(Xidx');
        r(j)= length(idx);
    end
    W = rand(d,k);
    iter = 1;
    
    %%  Optimization process
    while 1
        
        sumklw = 0;
        for j =1:c
            j1 = [1:j-1, j+1:c];
            for jj =1:c-1
                H{j,j1(jj)} = (mu(:,j) - mu(:,j1(jj)))*(mu(:,j) - mu(:,j1(jj)))';
                KLW(j,j1(jj)) = (1/2).*(log(det(W'*Sigma{j1(jj)}*W  + bias)) - log(det(W'*Sigma{j}*W + bias)) + trace(pinv(W'*Sigma{j1(jj)}*W)*(W'*(Sigma{j}+ H{j,j1(jj)})*W   )) );
                sumklw = sumklw + r(j)*r(j1(jj))*KLW(j,j1(jj));
                
            end
        end
        sumdaoKLW = zeros(d,k);
        for j =1:c
            j1 = [1:j-1, j+1:c];
            for jj =1:c-1
                G(j,j1(jj)) = r(j)*r(j1(jj))*KLW(j,j1(jj))/(sumklw);
                daoKLW{j,j1(jj)} = Sigma{j1(jj)}*W*pinv(W'*Sigma{j1(jj)}*W) - Sigma{j}*W*pinv(W'*Sigma{j}*W) + (Sigma{j}+ H{j,j1(jj)})*W*pinv(W'*Sigma{j1(jj)}*W) - Sigma{j1(jj)}*W*pinv(W'*Sigma{j1(jj)}*W)*W'*(Sigma{j}+ H{j,j1(jj)})*W*pinv(W'*Sigma{j1(jj)}*W);
                sumdaoKLW = sumdaoKLW + r(j)*r(j1(jj))*daoKLW{j,j1(jj)};
            end
        end
        for j =1:c
            j1 = [1:j-1, j+1:c];
            for jj =1:c-1
                daoG{j,j1(jj)} = (r(j)*r(j1(jj))*daoKLW{j,j1(jj)}./sumklw - (r(j)*r(j1(jj))*KLW(j,j1(jj))* sumdaoKLW)/(sumklw)^2);
            end
        end
        cellVector = cellfun(@(x) x(~isempty(x)), daoG, 'UniformOutput', false);
        cellVector = cat(2, cellVector{:});
        [nn,cc] = size(cellVector);
        vector = G(:);
        vector = vector(vector~=0);
        daof = zeros(d,k);
        sumln = 0;
        for i1 = 1:cc-1
            daof = daof + (cellVector(i1) - cellVector(i1+1)) *(exp(vector(i1) - vector(i1+1))*(2*(vector(i1) - vector(i1+1))))./(1 + exp(vector(i1) - vector(i1+1)));
            sumln = sumln + log(1 + exp(vector(i1) - vector(i1+1)));
        end
        
        %% updata W_v
        Wone = W + ones(d,k);
        derivative_of_rt = [];
        for i = 1:k
            Wlnnorm = sum(log(Wone(:,i)));
            dWln = 1./(log(Wone(:,i))+eps);
            Dj = Wlnnorm.*diag(dWln);
            derivative_of_rt(:,i) = 2.*lambda*Dj*W(:,i);
        end
        daof = abs(daof);
        % Solving Orthogonal Constraints
        [W] = OptStiefelGBB_YYQ(W, daof,derivative_of_rt, sumln, p,alpha, lambda,v);
        l(v) = sumln;
        sumw = 0;
        for i =1:k
            for j =1:d
                lnw(j) = log(1+abs(W(j,i)));
            end
            sumlnm = sum(lnw);
            sumw = sumw + sumlnm^2;
        end
        iter = iter + 1;
        if  iter==10,   break,   end
        
        
    end
    WW = cat(1,WW,W);
end
%% solving alpha
for v1 = 1:3
    alpha(v1) = (l(v1)^(1/(1-p)))/sum(power(l,1/(1-p)));
end

end
