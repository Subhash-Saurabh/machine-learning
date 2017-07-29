function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

a=size(z,1);
b=size(z,2);

for(i=1:a)
 for(j=1:b)
g(i,j)=1/(1+exp(-z(i,j)))

end
end
% =============================================================

end
