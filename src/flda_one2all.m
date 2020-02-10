function [rate, nambig] = flda_one2all(data, labels, test_data, test_labels, dmeasure, aheuristic, debug)
    % DEBUG FLAG
    DEBUG = debug;

    % use mean or gaussian to determine thresholds
    useMean = 0;
    useGauss = 0;

    % use a tree or prior
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
    C1_rest = data(find(labels ~= 1), :);
    C2_rest = data(find(labels ~= 2), :);
    C3_rest = data(find(labels ~= 3), :);
    C4_rest = data(find(labels ~= 4), :);
    C5_rest = data(find(labels ~= 5), :);

    % compute projection vectors
    w_1_all = flda2(C1, C1_rest);
    w_2_all = flda2(C2, C2_rest);
    w_3_all = flda2(C3, C3_rest);
    w_4_all = flda2(C4, C4_rest);
    w_5_all = flda2(C5, C5_rest);
    
    % collect all W vectors
    W_inds = [[1, 6]; [2, 6]; [3, 6]; [4, 6]; [5, 6]];
    W = [w_1_all, w_2_all, w_3_all, w_4_all, w_5_all];
    W_u = [[mean(C1 * w_1_all), mean(C1_rest * w_1_all)]; ...
           [mean(C2 * w_2_all), mean(C2_rest * w_2_all)]; ...
           [mean(C3 * w_3_all), mean(C3_rest * w_3_all)]; ...
           [mean(C4 * w_4_all), mean(C4_rest * w_4_all)]; ...
           [mean(C5 * w_5_all), mean(C5_rest * w_5_all)]];
    W_std = [[std(C1 * w_1_all), std(C1_rest * w_1_all)]; ...
             [std(C2 * w_2_all), std(C2_rest * w_2_all)]; ...
             [std(C3 * w_3_all), std(C3_rest * w_3_all)]; ...
             [std(C4 * w_4_all), std(C4_rest * w_4_all)]; ...
             [std(C5 * w_5_all), std(C5_rest * w_5_all)]];

    % DEBUG
    if DEBUG == 1
        DEBUG_plot_classes(C1, C2, C3, C4, C5);
        if useMean == 1
            DEBUG_plot_thresholds_mean(C1, C2, C3, C4, C5, C1_rest, C2_rest, C3_rest, C4_rest, C5_rest, W, W_u);
        elseif useGauss == 1
            DEBUG_plot_thresholds_gauss(C1, C2, C3, C4, C5, C1_rest, C2_rest, C3_rest, C4_rest, C5_rest, W, W_u, W_std);
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
        
        votes = zeros(num_classes+1, 1);
        for j = 1:numel(x_proj)
            if useMean == 1
                ind = meanCmp(x_proj(j), W_u(j, 1), W_u(j, 2), W_inds(j,:));
            elseif useGauss == 1
                ind = gaussCmp(x_proj(j), W_u(j, 1), W_std(j, 1), W_u(j, 2), W_std(j, 2), W_inds(j,:));
            end
            
            % increment votes for decision
            votes(ind) = votes(ind) + 1;
        end

        % end of voting, tally up
        [m, ind] = max(votes(1:num_classes));

        % if all votes are zero, then we don't know and ignore
        if m == 0
           num_ambig = num_ambig + 1;
           continue;
        end

        if numel(find(votes == m)) > 1       
            if useNone == 1
                ind = 6;
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
    rate = num_correct / (num_tests  - nambig);
end

function ind = meanRunoff(x_proj, W_u, W_inds)
    % project x into each w and find distance to each
    min_d = Inf;
    ind = 0;
    for i = 1:numel(x_proj)
        d1 = norm(x_proj(i) - W_u(i, 1));

        if d1 < min_d
            ind = W_inds(i, 1);
            min_d = d1;
        end
    end
end

function ind = decideTree(x_proj, W_u, W_std, W_inds, useMean, useGuass)
    w_inds = 1:5;

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

function DEBUG_plot_thresholds_mean(C1, C2, C3, C4, C5, C1_rest, C2_rest, C3_rest, C4_rest, C5_rest, W, W_u)
    DEBUG_plot_threshold(C1, C1_rest, W(:, 1), 0.5 * (W_u(1, 1) + W_u(1, 2)), 'class 1 versus rest');
    DEBUG_plot_threshold(C2, C2_rest, W(:, 2), 0.5 * (W_u(2, 1) + W_u(2, 2)), 'class 2 versus rest');
    DEBUG_plot_threshold(C3, C3_rest, W(:, 3), 0.5 * (W_u(3, 1) + W_u(3, 2)), 'class 3 versus rest');
    DEBUG_plot_threshold(C4, C4_rest, W(:, 4), 0.5 * (W_u(4, 1) + W_u(4, 2)), 'class 4 versus rest');
    DEBUG_plot_threshold(C5, C5_rest, W(:, 5), 0.5 * (W_u(5, 1) + W_u(5, 2)), 'class 5 versus rest');
end

function DEBUG_plot_thresholds_gauss(C1, C2, C3, C4, C5, C1_rest, C2_rest, C3_rest, C4_rest, C5_rest, W, W_u, W_std)
    DEBUG_plot_threshold(C1, C1_rest, W(:, 1), gaussIntersect(W_u(1, 1), W_std(1,1), W_u(1, 2), W_std(1, 2)), 'class 1 versus rest');
    DEBUG_plot_threshold(C2, C2_rest, W(:, 2), gaussIntersect(W_u(2, 1), W_std(2,1), W_u(2, 2), W_std(2, 2)), 'class 2 versus rest');
    DEBUG_plot_threshold(C3, C3_rest, W(:, 3), gaussIntersect(W_u(3, 1), W_std(3,1), W_u(3, 2), W_std(3, 2)), 'class 3 versus rest');
    DEBUG_plot_threshold(C4, C4_rest, W(:, 4), gaussIntersect(W_u(4, 1), W_std(4,1), W_u(4, 2), W_std(4, 2)), 'class 4 versus rest');
    DEBUG_plot_threshold(C5, C5_rest, W(:, 5), gaussIntersect(W_u(5, 1), W_std(5,1), W_u(5, 2), W_std(5, 2)), 'class 5 versus rest');
end

function DEBUG_plot_threshold(C1, C2, w, t, titleStr)
    figure;
    hax = axes;
    hold on;
    histfit(C1 * w); hold on; histfit(C2 * w);
    line([t t], get(hax, 'YLim'), 'Color', [1 0 0], 'LineWidth', 3);
    title(titleStr);
end
