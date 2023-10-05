function [Xnews] = ELMAE(X, ActivF, hidden_neurons, C)
% ELM-AE:the function  create an auto-encoder based ELM. 
% number_neurons:number of neurons in hidden layer.
% X: the training set.
input_weights = rand(hidden_neurons,size(X,2))*2-1;
input_weights=orth(input_weights');
bias = randn(hidden_neurons,1)*2-1;
bias = orth(bias');
% 2:calculating the hidden layer
tempH = X*input_weights;
tempH=tempH+bias';
% activation function
switch lower(ActivF)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1./(1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
% 3: calculate the output weights beta
HT=H'*H ; %H transposé produit H
tail=size(HT);
I = eye(tail(1))/C;
bet=inv(I+HT);
bet1=H'*X;
beta=bet*bet1;
% calculate the output : Unlike other networks the AEs uses the same weight
% beta as an input weigth for coding and output weights for decoding
% we will no longer use the old input weights:input_weights. 
Xnews=X*beta';
%betaX=X*beta';
%Xnew =double(tribas(Xnews));
%Xnews = 1+exp(-Xnews);
%Xnew=1./Xnews;
clear X;
clear input_weights;
clear bias;
clear tempH;
clear H;
clear beta;
clear betaX;

%output=Xnew*pinv(B');
% 4:calculate the prefomance
%prefomance=sqrt(mse(X-output));
end