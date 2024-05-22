
clear;clc;
%% Load Dataset
FileName = ['leaves.mat'];
load(FileName);
Xdata = X_100leaves;
Data = [Xdata{1}, Xdata{2}, Xdata{3}];
 Data = NormalizeFea(Data,0);
[m n] = size(Data);
Y = Y_100leaves;

%% Parameter settings
opt.nsel = n;  %Number of features
percentage = 0.7; %Percentage of selected features
opt.lambda = 1; %Adjustable hyperparameter lambda1
%% Ten fold cross validation to obtain training and testing sets
    ind(:,1) = crossvalind('Kfold',size(find(Y),1),10);
%% 10-fold cross validation results
for k = 1:10
    test = ind(:,1) == k;
    train = ~test;
    %% Calling the CIMVFS function
        [W1,~,alpha ] = CIMVFS(  Data(train,:),Y(train,:),opt );
%     [W1,theta,alpha ] = ANMVFS( Data(train,:),Y(train,:),opt );
    %% Using the obtained variables for feature selection
    theta = sqrt(sum(W1.*W1,2));
    [~, idx2] = sort(theta, 'descend');
    num = ceil(percentage*opt.nsel);
    theta(idx2(1:num-1)) = 1;
    theta(idx2(num:opt.nsel)) = 0;
    SelectFeaIdx = find(theta~=0); %Index of selected features
end



