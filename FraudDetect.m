T=readtable('creditcard.csv');
T.Class = categorical(T.Class);
Target = T.Class
%cvp=cvpartition(T,'Holdout',0.30);
CVO = cvpartition(T{:,:},'k',10);
err = zeros(CVO.NumTestSets,1);
%for i = 1:CVO.NumTestSets
    %trIdx = CVO.training(i);
    %teIdx = CVO.test(i);
    %ytest = classify(meas(teIdx,:),meas(trIdx,:),...
		 %species(trIdx,:));
    %err(i) = sum(~strcmp(ytest,species(teIdx)));
%end
cvErr = sum(err)/sum(CVO.TestSize);
data1_train = T(cvp.training, :);
data2_test = T(cvp.test, :);
clearvars -except data1_train data2_test

%number_neurons=i; %number of neurons
%ActivF='sig'; %activation function
%[Inputs,Targets,TsInputs,TsTargets]=SplitDB(trainRatio,T,Target)
%[prefomance,B,Hnew]=ELM_AE(Inputs,ActivF,number_neurons);
%regenerated=Hnew*pinv(B');
%subplot(122)
%imshow(regenerated);
%Tc=num2str(prefomance);
%Tc= ['RMSE = ' Tc];
%xlabel('regenerated')
%title(Tc)
%pause(0.25) 