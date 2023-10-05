
classdef extreme_learning_machine_classifier
    %EXTREME_LEARNING_MACHINE_CLASSIFIER Builds a classification extreme
    %learning machine. 
    
    properties
        Win = [];
        activation = {};
        Wout = [];
        functional = 0;
    end
    
    methods
        function obj = extreme_learning_machine_classifier(X,t,C,varargin)
            %EXTREME_LEARNING_MACHINE_CLASSIFIER Trains an ELM for
            %classification.
            % Inputs:
            %   X - Input data
            %   t - output data
            %   hidden - the number of hidden units (default: 10)
            %   activation - activation function (default: 'sigmoid')
            %                other values: 'tanh', 'relu', 'rbf', 'linear'
            p = inputParser;
            addRequired(p, 'X', @ismatrix);
            addRequired(p, 't', @isvector);
            addParameter(p, 'hidden', 10, @isnumeric);
            addParameter(p, 'activation', 'linear', @(x)any(validatestring(x,{'sigmoid','rbf','tanh','relu','linear'})));
            addParameter(p, 'functional', 0, @isnumeric);
            
            parse(p,1,1,varargin{:});
            obj.functional = p.Results.functional;
            
            switch p.Results.activation
                case 'sigmoid'
                    obj.activation = @(x)(1 ./ (1 + exp(-x)));
                case 'tanh'
                    obj.activation = @(x)(tanh(x));
                case 'relu'
                    obj.activation = @(x)(max(0,x));
                case 'rbf'
                    obj.activation = @(x)(radbas(x));
                case 'linear'
                    obj.activation = @(x)(x);
            end
            
            hidden = p.Results.hidden;
            dim=size(X);
            obj.Win = rand(dim(2),hidden);
            H = obj.activation (X*obj.Win);
            %obj.Wout = pinv(H)*transpose(t);
            HH=H'*H;
            tail=size(HH);
            I = (eye(tail)/C)+HH;
            in=inv(I);
            obj.Wout= in*H';
            obj.Wout=obj.Wout*t;
        end
        
        function [y,beta] = predict(obj,Xt,Ytest)
            H = obj.activation(Xt*obj.Win);
            y = H* obj.Wout;
            beta=pinv(H)*Ytest;
        end
        
    end
end
