dat=readtable('creditcard.csv');
dat.Class = categorical(dat.Class);
dat(:,30)=[];
dat(:,1)=[];
CVO = cvpartition(dat.Class,'kFold',5);
Accuracy_testing = rand(CVO.NumTestSets,1);
Accuracy_training = rand(CVO.NumTestSets,1);
Evaluation_train = zeros(CVO.NumTestSets,4);
Evaluation_test = zeros(CVO.NumTestSets,4);
mseError=rand(CVO.NumTestSets,1);
for i = 1:CVO.NumTestSets
    trIdx = dat(CVO.training(i), :);
    Ytr=trIdx.Class;
    trIdx.Class=[];
    trIdx=table2array(trIdx);
    Ytrain=double(Ytr);
    Id = zeros(size(Ytrain,1),1)+1;
    Ytrain=Ytrain-Id;
    row = find(Ytrain==0);
    trIdx=trIdx(row,:);
    Ytra=Ytrain(row);
    trId=transpose(trIdx);
    tesIdx = dat(CVO.test(i), :);
    Ytes=tesIdx.Class;
    Ytest=double(Ytes);
    Idt = zeros(size(Ytest,1),1)+1;
    Ytest=Ytest-Idt;
    tesIdx.Class=[];
    tesIdx=table2array(tesIdx);
    tesId=transpose(tesIdx);
    hiddenSize1 =10;   
    hiddenSize2 =20;
    autoenc = trainAutoencoder(trId,hiddenSize1);
    feat1=encode(autoenc,trId);
    autoenc1 = trainAutoencoder(feat1,hiddenSize2);
    feat2=encode(autoenc1,feat1);%input ELM
    stackednet = stack(autoenc,autoenc1);
    Ytri=transpose(Ytra);
    %% Train model and predict output
    mdl = extreme_learning_machine_classifier(transpose(feat2), Ytri); % Train ELM
    X_test = stackednet(tesId);
    y = mdl.predict(transpose(X_test)); % Predict
    % Print result
    Yt=double(y);
    fprintf("-------------------\n");
    fprintf("Model Acc.: %.2f%%\n", ...
    100 * sum(Yt == Ytest) / length(Ytest));
    fprintf("-------------------\n");
    mseError(i)=(100 * sum(Yt == Ytest) / length(Ytest));
    plotconfusion(Ytest,Yt);
end
clear data;