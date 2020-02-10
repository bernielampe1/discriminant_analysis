function w = flda2(C1, C2)

% Pass in two matricies, one for class 1 and one for class 2 then return
% the projection vector w. C1 and C2 must have one sample per row and
% the number of columns is the dimensionality of each sample and the
% number if rows is the total number of samples for each class.
% Assumes that number of samples in both C1 and C2 > than the
% dimensionality.

% compute dimensionality of each sample
d = size(C1, 2);

% check that the dimensionality is greater than the number of classes
assert(d >= 2);

% check to make sure that the dimensions of both classes is equal
assert(size(C1, 2) == size(C2, 2), 'Class samples have differing dimensionality');

% check to make sure that we have more samples than d
assert(d <= size(C1, 1), 'Class 1 number of columns/dimensions <= number of samples');
assert(d <= size(C2, 1), 'Class 2 number of columns/dimensions <= number of samples');

% compute S_b
m1 = mean(C1);
m2 = mean(C2);
S_b = (m1 - m2)' * (m1 - m2);

% compute S_w
S1 = zeros(d, d);
for i = 1:size(C1,1)
    x = C1(i, :);
    S1 = S1 + (x - m1)' * (x - m1);
end

S2 = zeros(d, d);
for i = 1:size(C2,1)
    x = C2(i, :);
    S2 = S2 + (x - m2)' * (x - m2);
end

S_w = S1 + S2;

% compute w
w = S_w \ (m1 - m2)';

% compute fishers quotient
%J = (w' * S_b * w) / (w' * S_w * w);

% DEBUG: plot the data samples of C1, C2 and the direction of w as a line
%wp = w / norm(w);
%if abs(wp(1)) > abs(wp(2))
%   x = min(min(C1(:,1), C2(:,1))):max(max(C1(:,1), C2(:,1)));
%    y = wp(2) / wp(1) .* x;
%else
%    y = min(min(C1(:,2), C2(:,2))):max(max(C1(:,2), C2(:,2)));
%    x = wp(1) / wp(2) .* y;
%end
%plot(C1(:,1), C1(:,2), 'b*', C2(:,1), C2(:, 2), 'r*', x, y, 'g', 'LineWidth', 3);

% DEBUG: project the samples from C1 and C2 and then histogram and plot threshold
%C1_proj = C1 * w;
%C2_proj = C2 * w;

%figure;
%hist(C2_proj); hold on; hist(C1_proj);

end
