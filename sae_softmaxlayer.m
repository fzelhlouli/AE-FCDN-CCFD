dat=readtable('creditcard.csv');
dat.Class = categorical(dat.Class);
dat(:,30)=[];
dat(:,1)=[];
%data.Class=double(data.Class);
%k=4;%4 ELM AE 
%C=100;
%hidden_neurons1=40;
%hidden_neurons2=40;
%activf='radbas';
%Crossvalidation of data 10 kfold
%indices = crossvalind('Kfold',A,10);
CVO = cvpartition(dat.Class,'kFold',5);
Accuracy_testing = rand(CVO.NumTestSets,1);
Accuracy_training = rand(CVO.NumTestSets,1);
Evaluation_train = zeros(CVO.NumTestSets,4);
Evaluation_test = zeros(CVO.NumTestSets,4);
mseError=rand(CVO.NumTestSets,1);
for i = 1:CVO.NumTestSets
    trIdx = dat(CVO.training(i), :);
    Ytr=trIdx.Class;
    Ytrain=double(Ytr);
    Id = zeros(size(Ytrain,1),1)+1;
    Ytrain=Ytrain-Id;
    trIdx.Class=[];
    trIdx=table2array(trIdx);
    %bodyfatNet = feedforwardnet(10);
    %bodyfatNet = train(bodyfatNet,trIdx,Ytr);
    %y = bodyfatNet(trIdx);
    trId=transpose(trIdx);
    tesIdx = dat(CVO.test(i), :);
    Ytes=tesIdx.Class;
    Ytest=double(Ytes);
    Idt = zeros(size(Ytest,1),1)+1;
    Ytest=Ytest-Idt;
    totl=sum(Ytest~=0)
    tesIdx.Class=[];
    tesIdx=table2array(tesIdx);
    tesId=transpose(tesIdx);
    %for p = 1:k
    %    Xnew_train = AE(trIdx,activf,hidden_neurons1,C);
    %    trIdx = Xnew_train;
    %end
    %for r = 1:k
    %    Xnew_test = AE(tesIdx,activf,hidden_neurons1,C);
    %    tesIdx = Xnew_test;
    %end
    %[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy, EVAL_train, EVAL_test] = ELMSAE(Xnew_train, Ytrain,Xnew_test,Ytest, hidden_neurons2, activf, C);
    %Accuracy_testing(i)= TestingAccuracy;
    %Accuracy_training(i)= TrainingAccuracy;
    %Evaluation_train(i,:)= EVAL_train;
    %Evaluation_test(i,:)= EVAL_test;
    hiddenSize1 =20;   
    hiddenSize2 =20;
    autoenc = trainAutoencoder(trId,hiddenSize1);
    feat1=encode(autoenc,trId);
    autoenc1 = trainAutoencoder(feat1,hiddenSize2);
    feat2=encode(autoenc1,feat1);
    Ytrai=transpose(Ytrain);
    softnet=trainSoftmaxLayer(feat2,Ytrai);
    stackednet=stack(autoenc,autoenc1,softnet);
    stackednet=train(stackednet,trId,Ytrai);
    y=stackednet(tesId);
    Ytes=transpose(Ytest);
    plotconfusion(Ytes,y)
    %calculer recall precision f1-score
end
clear data;