function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% Part 1.2

rTheta = theta(2:end);

hp = X*theta-y;

nonreg = hp'*hp / (2*m);
reg = lambda*( sum(rTheta).^2  )/(2*m);

J = nonreg + reg;

% Part 1.3

grad =  X'*hp / m;
grad(2:end) += lambda*rTheta/m;

% =========================================================================

grad = grad(:);

end
