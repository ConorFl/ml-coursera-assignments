function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%



J = 1 / m * sum(-1 * y .* log(sigmoid(X * theta)) -
				(1 - y) .* log(1 - sigmoid(X * theta)));

% Full answer for n x 1 vector: sum(sigmoid(X * theta - y) .* X).'
diffs = sigmoid(X * theta) - y; % n x 1 vector
mult_across_x = diffs .* X; 	% m x n matrix where each each element was mult by the row in diffs vector
sum_down = sum(mult_across_x);	% 1 x n row vector of all sums for each j in [1, n]

grad = 1 / m .* (sum_down.');	% Transpose sum_down to an n x 1 vector (this step seems optional)

% =============================================================

end
