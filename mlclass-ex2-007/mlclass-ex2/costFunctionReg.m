function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
tl = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


s = sigmoid(X*theta);
r_theta = theta(2:length(theta));

% J = sum ( -y .* log(s) .+ (1-y) .* log(1-s) ) / m +lambda * sum(theta(2,length(theta)))/(2*m);

t = y .* log(s) .+ (1-y) .* log(1-s);
J = (-1/m) * sum( t ) + lambda * ( sum( r_theta.^2 ) )/2/m;


grad = ((s .- y)' * X)' / m .+ lambda/m * theta;

grad(1) =  (s .- y)' * X(:,1) / m;


% =============================================================

end
