dat=readtable('creditcard.csv');%optimisation of w et b
dat.Class = categorical(dat.Class);
dat(:,30)=[];
dat(:,1)=[];
CVO = cvpartition(dat.Class,'kFold',5);
Accuracy_testing = rand(CVO.NumTestSets,1);
Accuracy_training = rand(CVO.NumTestSets,1);
Evaluation_train = zeros(CVO.NumTestSets,4);
Evaluation_test = zeros(CVO.NumTestSets,4);
mseError=rand(CVO.NumTestSets,1);
hiddenSize1 =10;   
hiddenSize2 =10;
for i = 1:CVO.NumTestSets
    trIdx = dat(CVO.training(i), :);
    Ytr=trIdx.Class;
    trIdx.Class=[];
    trIdx=table2array(trIdx);
    Ytrain=double(Ytr);
    Id = zeros(size(Ytrain,1),1)+1;
    Ytrain=Ytrain-Id;
    trId=transpose(trIdx);
    tesIdx = dat(CVO.test(i), :);
    Ytes=tesIdx.Class;
    Ytest=double(Ytes);
    Idt = zeros(size(Ytest,1),1)+1;
    Ytest=Ytest-Idt;
    tesIdx.Class=[];
    tesIdx=table2array(tesIdx);
    tesId=transpose(tesIdx);
    autoenc = trainAutoencoder(trId,hiddenSize1);
    feat1=encode(autoenc,trId);
    autoenc1 = trainAutoencoder(feat1,hiddenSize2);
    feat2=encode(autoenc1,feat1);%input ELM
    stackednet = stack(autoenc,autoenc1);
    Ytri=transpose(Ytra);
    C=0.01;%Regularization value to compute Beta
    %% Train model and predict output
    mdl = extreme_learning_machine_classifier(transpose(feat2), Ytrain,C); % Train ELM
    X_test = stackednet(tesId);
    mdl.layers{2}.transferFcn = 'softmax';
    [y,beta] = mdl.predict(transpose(X_test),Ytest); % Predict
    [W,b] = optimization(Ytest,beta);
    % Print result
    fprintf("-------------------\n");
    fprintf("Model Acc.: %.2f%%\n", ...
    100 * sum(y == Ytest) / length(Ytest));
    fprintf("-------------------\n");
    mseError(i)=(100 * sum(y == Ytest) / length(Ytest));
    plotconfusion(Ytest,y);
end
clear data;