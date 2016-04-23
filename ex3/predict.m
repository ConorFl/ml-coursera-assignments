function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% x_col_count = size(X, 2)
% x_row_count = size(X, 1)
% theta_1_col_count = size(Theta1, 2)
% theta_1_row_count = size(Theta1, 1)

X = [ones(m, 1) X];

z_2 = X * Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1) a_2];


% z_2_col_count = size(z_2, 2)
% z_2_row_count = size(z_2, 1)
% theta_2_col_count = size(Theta2, 2)
% theta_2_row_count = size(Theta2, 1)

z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);
% a_3_col_count = size(a_3, 2)
% a_3_row_count = size(a_3, 1)

[a, p] = max(a_3, [], 2);

% =========================================================================


end
