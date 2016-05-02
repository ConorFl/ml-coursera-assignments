function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
A_1 = [ones(m, 1) X]; % Add ones
Z_2 = A_1 * Theta1';
A_2 = sigmoid(Z_2);
a_2_row_count = size(A_2, 1);
A_2 = [ones(a_2_row_count, 1) A_2]; % Add ones
Z_3 = A_2 * Theta2';
A_3 = sigmoid(Z_3);

for i = 1:m
	pred_vect = A_3(i,:)';
	y_vect = (1:num_labels)' == y(i);
	costs = -1 * y_vect .* log(pred_vect) - (1 - y_vect) .* log(1 - pred_vect);
	J = J + sum(costs);
end

J = J / m;
Theta1_without_bias = Theta1; % Exclude bias term from regularization
Theta1_without_bias(:, 1) = [];
Theta2_without_bias = Theta2; % Exclude bias term from regularization
Theta2_without_bias(:, 1) = [];

regTerm = lambda / (2 * m) * (sum(Theta1_without_bias(:) .^ 2) + sum(Theta2_without_bias(:) .^ 2));

J = J + regTerm;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% Use A_3 already computed in Part 1

Y_Matrix = [];
for i = 1:m
  Y_Matrix = [Y_Matrix; (1:num_labels == y(i))];
end

D_3 = A_3 - Y_Matrix;

D_2 = D_3 * Theta2;
D_2 = (D_2(:, 2:end)) .* sigmoidGradient(Z_2);

Theta2_grad = (1 / m) * D_3' * A_2;

Theta1_grad = (1 / m) * D_2' * A_1;

% Failed/confused attempt one by one
% for i = 1:m
%   pred_vect = A_3(i, :)';
%   z_2 = Z_2(i, :)';
%   y_vect = (1:num_labels)' == y(i);
%   d_3 = pred_vect - y_vect;

%   % d_2 = (Theta2' * d_3)(2:end);
%   % Remove bias element
%   d_2 = (Theta2' * d_3) .* sigmoidGradient(Z_2);

% end

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2_reg_factor = lambda / m * Theta2
Theta2_reg_factor(:, 1) = 0; % Don't regularize bias term
Theta2_grad = Theta2_grad + Theta2_reg_factor;

Theta1_reg_factor = lambda / m * Theta1;
Theta1_reg_factor(:, 1) = 0; % Don't regularize bias term
Theta1_grad = Theta1_grad + Theta1_reg_factor;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
