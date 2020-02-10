function [rate, nambig] = flda_one2one(data, labels, test_data, test_labels, dmeasure, aheuristic, debug)
    % DEBUG FLAG
    DEBUG = debug;
    
    % use mean or gaussian to determine thresholds
    useMean = 0;
    useGauss = 0;
    
    % ambiguity resolution strategy
    useNone = 0;
    usePrior = 0;
    useTree = 0;
    useMeanRunoff = 0;
    
    if strcmp(dmeasure, 'mean')
        useMean = 1;
    elseif strcmp(dmeasure, 'gauss')
        useGauss = 1;
    end
    
    if strcmp(aheuristic, 'none')
        useNone = 1;
    elseif strcmp(aheuristic, 'prior')
        usePrior = 1;
    elseif strcmp(aheuristic, 'tree')
        useTree = 1;
    elseif strcmp(aheuristic, 'mean')
        useMeanRunoff = 1;
    end

    % get all the classes
    C1 = data(find(labels == 1), :);
    C2 = data(find(labels == 2), :);
    C3 = data(find(labels == 3), :);
    C4 = data(find(labels == 4), :);
    C5 = data(find(labels == 5), :);
    
    % compute projection vectors
    w_1_2 = flda2(C1, C2);
    w_1_3 = flda2(C1, C3);
    w_1_4 = flda2(C1, C4);
    w_1_5 = flda2(C1, C5);
    w_2_3 = flda2(C2, C3);
    w_2_4 = flda2(C2, C4);
    w_2_5 = flda2(C2, C5);
    w_3_4 = flda2(C3, C4);
    w_3_5 = flda2(C3, C5);
    w_4_5 = flda2(C4, C5);
    
    % collect all W vectors
    W = [w_1_2, w_1_3, w_1_4, w_1_5, w_2_3, w_2_4, w_2_5, w_3_4, w_3_5, w_4_5];
    W_inds = [[1, 2]; [1, 3]; [1, 4]; [1, 5]; [2, 3]; ...
              [2, 4]; [2, 5]; [3, 4]; [3, 5]; [4, 5]];
    W_u = [[mean(C1 * w_1_2), mean(C2 * w_1_2)]; ...
           [mean(C1 * w_1_3), mean(C3 * w_1_3)]; ...
           [mean(C1 * w_1_4), mean(C4 * w_1_4)]; ...
           [mean(C1 * w_1_5), mean(C5 * w_1_5)]; ...
           [mean(C2 * w_2_3), mean(C3 * w_2_3)]; ...
           [mean(C2 * w_2_4), mean(C4 * w_2_4)]; ...
           [mean(C2 * w_2_5), mean(C5 * w_2_5)]; ...
           [mean(C3 * w_3_4), mean(C4 * w_3_4)]; ...
           [mean(C3 * w_3_5), mean(C5 * w_3_5)]; ...
           [mean(C4 * w_4_5), mean(C5 * w_4_5)]];
    W_std = [[std(C1 * w_1_2), std(C2 * w_1_2)]; ...
             [std(C1 * w_1_3), std(C3 * w_1_3)]; ...
             [std(C1 * w_1_4), std(C4 * w_1_4)]; ...
             [std(C1 * w_1_5), std(C5 * w_1_5)]; ...
             [std(C2 * w_2_3), std(C3 * w_2_3)]; ...
             [std(C2 * w_2_4), std(C4 * w_2_4)]; ...
             [std(C2 * w_2_5), std(C5 * w_2_5)]; ...
             [std(C3 * w_3_4), std(C4 * w_3_4)]; ...
             [std(C3 * w_3_5), std(C5 * w_3_5)]; ...
             [std(C4 * w_4_5), std(C5 * w_4_5)]];
         
    if DEBUG == 1
        DEBUG_plot_classes(C1, C2, C3, C4, C5);
        if useMean == 1
            DEBUG_plot_thresholds_mean(C1, C2, C3, C4, C5, W, W_u);
        else
            DEBUG_plot_thresholds_gauss(C1, C2, C3, C4, C5, W, W_u, W_std);
        end
    end
        
    % run tests
    num_correct = 0;
    num_ambig = 0;
    num_tests = numel(test_labels);
    num_classes = numel(unique(labels));
    for i = 1:num_tests
        % generate random sample with parameter p
        p_i = test_labels(i);
        x_proj = test_data(i, :) * W;
        
        votes = zeros(num_classes, 1);
        for j = 1:size(W,2)
            if useMean == 1
                ind = meanCmp(x_proj(j), W_u(j, 1), W_u(j, 2), W_inds(j,:));
            elseif useGauss == 1
                ind = gaussCmp(x_proj(j), W_u(j, 1), W_std(j, 1), W_u(j, 2), W_std(j, 2), W_inds(j,:));
            end
            
            % increment votes for decision
            votes(ind) = votes(ind) + 1;
        end

        % end of voting, tally up
        [m, ind] = max(votes);

        if numel(find(votes == m)) > 1
            num_ambig = num_ambig + 1;
            
            if useNone == 1
                ind = 0;
            elseif usePrior == 1
                ind = max(find(votes == m));
            elseif useTree == 1
                ind = decideTree(x_proj, W_u, W_std, W_inds, useMean, useGauss);
            elseif useMeanRunoff == 1
                ind = meanRunoff(x_proj, W_u, W_inds);
            end
        end
        
        if ind == p_i
            num_correct = num_correct + 1;
        end
    end

    nambig = num_ambig;
    if useNone == 1
        rate = num_correct / (num_tests  - nambig);
    else
        rate = num_correct / num_tests;
    end
end

function ind = meanRunoff(x_proj, W_u, W_inds)
    % project x into each w and find distance to each
    min_d = Inf;
    ind = 0;
    for i = numel(x_proj)
        d1 = norm(x_proj(i) - W_u(i, 1));
        d2 = norm(x_proj(i) - W_u(i, 2));

        if d1 < min_d
            ind = W_inds(i, 1);
            min_d = d1;
        end
        
        if d2 < min_d
            ind = W_inds(i, 2);
            min_d = d2;
        end
    end
end

function ind = decideTree(x_proj, W_u, W_std, W_inds, useMean, useGuass)
    w_inds = [1, 5, 8, 10];

    for i = w_inds
       if useMean == 1
          ind = meanCmp(x_proj(i), W_u(i, 1), W_u(i, 2), W_inds(i, :)); 
       elseif useGuass == 1
          ind = gaussCmp(x_proj(i), W_u(i, 1), W_std(i, 1), W_u(i, 2), W_std(i, 2), W_inds(i, :));
       end
       if ind == W_inds(i, 1)
           return;
       end
    end
end

function ind = meanCmp(x, u1, u2, inds)
    d1 = norm(x - u1);
    d2 = norm(x - u2);
    if d1 < d2
        ind = inds(1);
    else
        ind = inds(2);
    end
end

function ind = gaussCmp(x, u1, s1, u2, s2, inds)
    p1 = normpdf(x, u1, s1);
    p2 = normpdf(x, u2, s2);
    if p1 > p2
        ind = inds(1);
    else
        ind = inds(2);
    end
end

function DEBUG_plot_classes(C1, C2, C3, C4, C5)
    figure;
    histfit(sum(C1,2));
    hold on;
    histfit(sum(C2,2));
    histfit(sum(C3,2));
    histfit(sum(C4,2));
    histfit(sum(C5,2));
    title('histograms of number of heads per class');
end

function DEBUG_plot_thresholds_mean(C1, C2, C3, C4, C5, W, W_u)
    DEBUG_plot_threshold(C1, C2, W(:, 1), 0.5 * (W_u(1, 1) + W_u(1, 2)), 'class 1 versus class 2');
    DEBUG_plot_threshold(C1, C3, W(:, 2), 0.5 * (W_u(2, 1) + W_u(2, 2)), 'class 1 versus class 3');
    DEBUG_plot_threshold(C1, C4, W(:, 3), 0.5 * (W_u(3, 1) + W_u(3, 2)), 'class 1 versus class 4');
    DEBUG_plot_threshold(C1, C5, W(:, 4), 0.5 * (W_u(4, 1) + W_u(4, 2)), 'class 1 versus class 5');
    DEBUG_plot_threshold(C2, C3, W(:, 5), 0.5 * (W_u(5, 1) + W_u(5, 2)), 'class 2 versus class 3');
    DEBUG_plot_threshold(C2, C4, W(:, 6), 0.5 * (W_u(6, 1) + W_u(6, 2)), 'class 2 versus class 4');
    DEBUG_plot_threshold(C2, C5, W(:, 7), 0.5 * (W_u(7, 1) + W_u(7, 2)), 'class 2 versus class 5');
    DEBUG_plot_threshold(C3, C4, W(:, 8), 0.5 * (W_u(8, 1) + W_u(8, 2)), 'class 3 versus class 4');
    DEBUG_plot_threshold(C3, C5, W(:, 9), 0.5 * (W_u(9, 1) + W_u(9, 2)), 'class 3 versus class 5');
    DEBUG_plot_threshold(C4, C5, W(:, 10), 0.5 * (W_u(10, 1) + W_u(10, 2)), 'class 4 versus class 5');
end

function DEBUG_plot_thresholds_gauss(C1, C2, C3, C4, C5, W, W_u, W_std)
    DEBUG_plot_threshold(C1, C2, W(:, 1), gaussIntersect(W_u(1,1), W_std(1, 1), W_u(1, 2), W_std(1, 2)), 'class 1 versus class 2');
    DEBUG_plot_threshold(C1, C3, W(:, 2), gaussIntersect(W_u(2,1), W_std(2, 1), W_u(2, 2), W_std(2, 2)), 'class 1 versus class 3');
    DEBUG_plot_threshold(C1, C4, W(:, 3), gaussIntersect(W_u(3,1), W_std(3, 1), W_u(3, 2), W_std(3, 2)), 'class 1 versus class 4');
    DEBUG_plot_threshold(C1, C5, W(:, 4), gaussIntersect(W_u(4,1), W_std(4, 1), W_u(4, 2), W_std(4, 2)), 'class 1 versus class 5');
    DEBUG_plot_threshold(C2, C3, W(:, 5), gaussIntersect(W_u(5,1), W_std(5, 1), W_u(5, 2), W_std(5, 2)), 'class 2 versus class 3');
    DEBUG_plot_threshold(C2, C4, W(:, 6), gaussIntersect(W_u(6,1), W_std(6, 1), W_u(6, 2), W_std(6, 2)), 'class 2 versus class 4');
    DEBUG_plot_threshold(C2, C5, W(:, 7), gaussIntersect(W_u(7,1), W_std(7, 1), W_u(7, 2), W_std(7, 2)), 'class 2 versus class 5');
    DEBUG_plot_threshold(C3, C4, W(:, 8), gaussIntersect(W_u(8,1), W_std(8, 1), W_u(8, 2), W_std(8, 2)), 'class 3 versus class 4');
    DEBUG_plot_threshold(C3, C5, W(:, 9), gaussIntersect(W_u(9,1), W_std(9, 1), W_u(9, 2), W_std(9, 2)), 'class 3 versus class 5');
    DEBUG_plot_threshold(C4, C5, W(:, 10), gaussIntersect(W_u(10,1), W_std(10, 1), W_u(10, 2), W_std(10, 2)), 'class 4 versus class 5');
end

function DEBUG_plot_threshold(C1, C2, w, t, titleStr)
    figure;
    hax = axes;
    hold on;
    histfit(C1 * w); hold on; histfit(C2 * w);
    line([t t], get(hax, 'YLim'), 'Color', [1 0 0], 'LineWidth', 3);
    title(titleStr);
end
