function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy, EVAL_train, EVAL_test] = ELMSAE(X,Ytrain, Ynew, Ytest, NumberofHiddenNeurons, ActivF, C)

%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeigh=randn(NumberofHiddenNeurons,size(X,2));
InputWeight=orth(InputWeigh');
BiasofHiddenNeurons=randn(NumberofHiddenNeurons,1);
BiasofHiddenNeurons=orth(BiasofHiddenNeurons');

%%%%%%%%%%% Calculate hidden neuron output matrix H
tempHH = X*InputWeight;
tempHH=tempHH+BiasofHiddenNeurons';

switch lower(ActivF)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H=1+exp(-tempHH);
        H=1./H;
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempHH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempHH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempHH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempHH);
        %%%%%%%% More activation functions can be added here                
end
clear tempHH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
clear InputWeight;
clear BiasofHiddenNeurons;
%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
%HT=H'*H ; %H transposé produit H
%tail=size(HT);
%I = eye(tail(1))/C;
%bet=inv(I+HT);
Id = zeros(size(Ytrain,1),1)+1;
Ytrain=Ytrain-Id;
%bet1=H'*Ytrai;
Output=pinv(H)*Ytrain;
%HTes=H'*H ; %H transposé produit H
%tail=size(HTes);
%I = eye(tail)/C;
%bet=inv(I+HTes);
%bet1=H'*Ytrai;
%Output=bet*bet1;

end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=H*Output;                   %   Y: the actual output of the training data
clear H;
%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
Weigh=randn(NumberofHiddenNeurons,size(Ynew,2));
Weights=orth(Weigh');
BiasofHidden=randn(NumberofHiddenNeurons,1);
BiasofHidden=orth(BiasofHidden');
tempH_test = Ynew*Weights;
tempH_test=tempH_test+BiasofHidden';

clear test_data; 
clear Xnew;
clear Ytr;
clear Ytes;

switch lower(ActivF)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1./(1+exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = double(hardlim(tempH_test));        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=H_test*Output;                 %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
Idt = zeros(size(Ytest,1),1)+1;
Ytest=Ytest-1;

%%%%%% Calculate training & testing classification accuracy with function
EVAL_train = Evaluate(Ytrain, Y);
EVAL_test = Evaluate(Ytest, TY);

%%%%%%%%%% Calculate training & testing classification accuracy manualy

MissClassificationRate_Training=0;
MissClassificationRate_Testing=0;
TrainingAccuracy= 0;
TestingAccuracy= 0;

for s = 1 : size(X,1)
    if Ytrain(s)==Y(s)
       MissClassificationRate_Training=MissClassificationRate_Training+1;
    end
end
TrainingAccuracy=MissClassificationRate_Training/size(X,1);

for j = 1 : size(Ynew,1)
    if Ytest(j)==TY(j)
       MissClassificationRate_Testing=MissClassificationRate_Testing+1;
    end
end
TestingAccuracy=MissClassificationRate_Testing/size(Ynew,1);

