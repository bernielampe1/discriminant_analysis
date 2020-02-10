function [W, J] = flda(data, labels)

% Pass in a matrix "data" where each row is a sample and the number of
% columns is the dimensionality of each sample and the number of rows is
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

% compute S_w
S_w = zeros(d, d);
for i = 1:k
    inds = find(labels == classes(i));
    m_i = mean(data(inds, :));

    S_i = zeros(d, d);
    for j = 1:numel(inds)
        S_i = S_i + (data(inds(j), :) - m_i)' * (data(inds(j), :) - m_i);
    end
    S_w = S_w + S_i;
end

% compute S_b
S_b = zeros(d, d);
m = mean(data);
for i = 1:k
    inds = find(labels == classes(i));
    m_i = mean(data(inds, :));
    N_k = numel(inds);
    S_b = S_b + N_k .* (m_i - m)' * (m_i - m);
end

% compute eigenvectors of S_w^-1 * S_B
[V, D] = eig(inv(S_w) * S_b);

% compute the fishers quotient
J = det(V' * S_b * V) / det(V' * S_w * V);

% loop over the nonzero eigenvectors and assign to W
W = zeros(d, k-1);
j = 1;
for i = 1:length(V)
    if norm(D(:, i)) > 0.000001
        W(:, j) = V(:, i);
        j = j + 1;
    end
end

% DEBUG: project the samples from C1 and C2 and then histogram and plot threshold
%cmap = hsv(4);
%for i = 1:k
%    C_proj = data(labels == classes(i), :) * W;
%    plot(C_proj(:,1), 's', 'Color', cmap(i,:));
%    hold on;
%end

end
