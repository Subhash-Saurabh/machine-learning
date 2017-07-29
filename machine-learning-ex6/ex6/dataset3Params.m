function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma =3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
k=1;
set=zeros(1,64);
sig=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
c=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
C=0.01
for i=1:8
    for j=1:8
  
        
        sigma=sig(i);
        C=c(j);
  model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
  predictions = svmPredict(model, Xval);
  err=norm(mean(double(predictions ~= yval)));
  
  set(k)=err;
  k=k+1;
  
    end
  
end
for k=1:64
    if set(k)<10^(-3)
        set(k)=10;
    end
end
[x,I]=min(set);

C=c(1+mod(I,8))
sigma=sig(1+floor(I/8))

        

% =========================================================================

end
