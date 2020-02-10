function rate = flda_proj_classify(data, labels, test_data, test_labels, dmeasure, debug)
    % DEBUG FLAG
    DEBUG = debug;

    % use mean or gaussian to determine thresholds
    useMean = 0;
    useGauss = 0;
    
    if strcmp(dmeasure, 'mean')
        useMean = 1;
    elseif strcmp(dmeasure, 'gauss')
        useGauss = 1;
    end
   
    % get all the classes
    C1 = data(find(labels == 1), :);
    C2 = data(find(labels == 2), :);
    C3 = data(find(labels == 3), :);
    C4 = data(find(labels == 4), :);
    C5 = data(find(labels == 5), :);
    
    % DEBUG
    if DEBUG == 1
        figure;
        histfit(sum(C1,2));
        hold on;
        histfit(sum(C2,2));
        histfit(sum(C3,2));
        histfit(sum(C4,2));
        histfit(sum(C5,2));
        title('histograms of number of heads per class');
    end
    
    % find the projection vectors
    W = flda_proj(data, labels);
        
    % change here to use different number of vectors
    % W = W(:,1);
    
    C1_proj = C1 * W;
    C2_proj = C2 * W;
    C3_proj = C3 * W;
    C4_proj = C4 * W;
    C5_proj = C5 * W;
    
    % compute threshold parameters
    C1_proj_u = mean(C1_proj); C1_proj_s = cov(C1_proj);
    C2_proj_u = mean(C2_proj); C2_proj_s = cov(C2_proj);
    C3_proj_u = mean(C3_proj); C3_proj_s = cov(C3_proj);
    C4_proj_u = mean(C4_proj); C4_proj_s = cov(C4_proj);
    C5_proj_u = mean(C5_proj); C5_proj_s = cov(C5_proj);

    % run tests
    num_correct = 0;
    num_tests = numel(test_labels);
    for i = 1:num_tests
        % generate random sample with parameter p
        p_i = test_labels(i);
        x_proj = test_data(i, :) * W;
        
        % compute minmum of distance from proj(x) to projected class means
        if useMean == 1
            dists = zeros(5, 1);
            dists(1) = norm(x_proj - C1_proj_u);
            dists(2) = norm(x_proj - C2_proj_u);
            dists(3) = norm(x_proj - C3_proj_u);
            dists(4) = norm(x_proj - C4_proj_u);
            dists(5) = norm(x_proj - C5_proj_u);
            [m, ind] = min(dists);
        elseif useGauss == 1
            probs = zeros(5, 1);
            probs(1) = mvnpdf(x_proj, C1_proj_u, C1_proj_s);
            probs(2) = mvnpdf(x_proj, C2_proj_u, C2_proj_s);
            probs(3) = mvnpdf(x_proj, C3_proj_u, C3_proj_s);
            probs(4) = mvnpdf(x_proj, C4_proj_u, C4_proj_s);
            probs(5) = mvnpdf(x_proj, C5_proj_u, C5_proj_s);
            [m, ind] = max(probs);
        end
        
        if ind == p_i
            num_correct = num_correct + 1;
        end
    end

    rate = num_correct / num_tests;
end
