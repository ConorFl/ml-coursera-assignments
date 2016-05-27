function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:size(X,1)
	% Get row vector of element x_i
	x = X(i, :);
	% Subtract x from ALL centroids (Octave magic)
	distances = centroids - x;
	% Square values
	distances = distances .^ 2;
	% then sum for distance
	distancesV = sum(distances, 2);
	% Find min distance (technically a sq root should be used for distance 
	% but it doesn't matter since we're find the min)
	[val, index] = min(distancesV);
	idx(i) = index;

end


% =============================================================

end

