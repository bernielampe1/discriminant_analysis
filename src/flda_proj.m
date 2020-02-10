function [W, J] = flda_proj(data, labels)

% Pass in a matrix "data" where each row is a sample and the number of
% columns is the dimensinoality of each sample and the number of rows is
% the total number of samples. Assume that the number of samples for
% each class is > the dimensionality. Also, pass in a vector label where
% the entry is the class label for the sample in the i'th row of the data
% matrix.

% compute dimensionality of samples
d = size(data, 2);

% find number of classes
classes = unique(labels);
k = numel(classes);

% check that the dimensionality is greater than the number of classes
assert(d >= k);

% make sure we have more samples than dimensions for each class
for i = 1:k
    assert(d <= size(find(labels == classes(i)), 1), 'One class does not have enough samples');
end

W = zeros(d, k-1);
for i = 1:k-1
   c1_inds = find(labels == i);
   c2_inds = find(labels > i);
   w = flda2(data(c2_inds, :), data(c1_inds, :));
   W(:, i) = w;
   P_w = eye(d,d) - w*inv(w'*w)*w';
   data = data * P_w;
end

end
