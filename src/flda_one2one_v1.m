function [rate, nambig] = flda_one2one()
    % DEBUG FLAG
    DEBUG = 1;

    % number of new data samples to generate
    num_tests = 1000;
    
    % dimension of each vector
    d = 100;
    
    % data generation factor
    factor = 200;
    
    % use mean or gaussian to determine thresholds
    useMean = 1;
    useGauss = 0;
    
    % ambiguity resolving
    useNone = 1;
    usePrior = 0;
    useRunnoff = 0;
    useTree = 0;
    
    % probabilities of the coins
    coins = [0.1, 0.2, 0.3, 0.4, 0.5];

    % generate data with d =  100
    data = genDelta(coins, d, factor);
    
    % get all the classes
    s = factor;
    C1 = data(1:s, :);
    C2 = data(1*s+1:3*s, :);
    C3 = data(3*s+1:6*s, :);
    C4 = data(6*s+1:10*s, :);
    C5 = data(10*s+1:15*s, :);
    
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
    
    % compute thresholds
    if useMean == 1
        t_1_2 = 0.5 * (mean(C1 * w_1_2) + mean(C2 * w_1_2));
        t_1_3 = 0.5 * (mean(C1 * w_1_3) + mean(C3 * w_1_3));
        t_1_4 = 0.5 * (mean(C1 * w_1_4) + mean(C4 * w_1_4));
        t_1_5 = 0.5 * (mean(C1 * w_1_5) + mean(C5 * w_1_5));
        t_2_3 = 0.5 * (mean(C2 * w_2_3) + mean(C3 * w_2_3));
        t_2_4 = 0.5 * (mean(C2 * w_2_4) + mean(C4 * w_2_4));
        t_2_5 = 0.5 * (mean(C2 * w_2_5) + mean(C5 * w_2_5));
        t_3_4 = 0.5 * (mean(C3 * w_3_4) + mean(C4 * w_3_4));
        t_3_5 = 0.5 * (mean(C3 * w_3_5) + mean(C5 * w_3_5));
        t_4_5 = 0.5 * (mean(C4 * w_4_5) + mean(C5 * w_4_5));
    elseif useGauss == 1
        t_1_2 = gaussIntersect(mean(C1 * w_1_2), std(C1 * w_1_2), mean(C2 * w_1_2), std(C2 * w_1_2));
        t_1_3 = gaussIntersect(mean(C1 * w_1_3), std(C1 * w_1_3), mean(C3 * w_1_3), std(C3 * w_1_3));
        t_1_4 = gaussIntersect(mean(C1 * w_1_4), std(C1 * w_1_4), mean(C4 * w_1_4), std(C4 * w_1_4));
        t_1_5 = gaussIntersect(mean(C1 * w_1_5), std(C1 * w_1_5), mean(C5 * w_1_5), std(C5 * w_1_5));
        t_2_3 = gaussIntersect(mean(C2 * w_2_3), std(C2 * w_2_3), mean(C3 * w_2_3), std(C3 * w_2_3));
        t_2_4 = gaussIntersect(mean(C2 * w_2_4), std(C2 * w_2_4), mean(C4 * w_2_4), std(C4 * w_2_4));
        t_2_5 = gaussIntersect(mean(C2 * w_2_5), std(C2 * w_2_5), mean(C5 * w_2_5), std(C5 * w_2_5));
        t_3_4 = gaussIntersect(mean(C3 * w_3_4), std(C3 * w_3_4), mean(C4 * w_3_4), std(C4 * w_3_4));
        t_3_5 = gaussIntersect(mean(C3 * w_3_5), std(C3 * w_3_5), mean(C5 * w_3_5), std(C5 * w_3_5));
        t_4_5 = gaussIntersect(mean(C4 * w_4_5), std(C4 * w_4_5), mean(C5 * w_4_5), std(C5 * w_4_5));
    end
    
    % DEBUG
    if DEBUG == 1
        DEBUG_plot_threshold(C1, C2, w_1_2, t_1_2, 'Class 1 and 2 Projected with Threshold');
        DEBUG_plot_threshold(C1, C3, w_1_3, t_1_3, 'Class 1 and 3 Projected with Threshold');
        DEBUG_plot_threshold(C1, C4, w_1_4, t_1_4, 'Class 1 and 4 Projected with Threshold');
        DEBUG_plot_threshold(C1, C5, w_1_5, t_1_5, 'Class 1 and 5 Projected with Threshold');
        DEBUG_plot_threshold(C2, C3, w_2_3, t_2_3, 'Class 2 and 3 Projected with Threshold');
        DEBUG_plot_threshold(C2, C4, w_2_4, t_2_4, 'Class 2 and 4 Projected with Threshold');
        DEBUG_plot_threshold(C2, C5, w_2_5, t_2_5, 'Class 2 and 5 Projected with Threshold');
        DEBUG_plot_threshold(C3, C4, w_3_4, t_3_4, 'Class 3 and 4 Projected with Threshold');
        DEBUG_plot_threshold(C3, C5, w_3_5, t_3_5, 'Class 3 and 5 Projected with Threshold');
        DEBUG_plot_threshold(C4, C5, w_4_5, t_4_5, 'Class 4 and 5 Projected with Threshold');
    end
        
    % run tests
    num_correct = 0;
    num_ambig = 0;
    for i = 1:num_tests
        % choose random index into coins array
        p_i = randi([1,5]);
        p = coins(p_i);
        
        % generate random sample with parameter p
        x = genObservation(d, p);
        votes = zeros(numel(coins), 1);
        
        % test class c1 versus c2        
        if x * w_1_2 > t_1_2
            votes(1) = votes(1) + 1;
        else
            votes(2) = votes(2) + 1;
        end
        
        % test class c1 versus c3
        if x * w_1_3 > t_1_3
            votes(1) = votes(1) + 1;
        else
            votes(3) = votes(3) + 1;
        end
        
        % test class c1 versus c4
        if x * w_1_4 > t_1_4
            votes(1) = votes(1) + 1;
        else
            votes(4) = votes(4) + 1;
        end
        
        % test class c1 versus c5
        if x * w_1_5 > t_1_5
            votes(1) = votes(1) + 1;
        else
            votes(5) = votes(5) + 1;
        end
        
        % test class c2 versus c3
        if x * w_2_3 > t_2_3
            votes(2) = votes(2) + 1;
        else
            votes(3) = votes(3) + 1;
        end
        
        % test class c2 versus c4
        if x * w_2_4 > t_2_4
            votes(2) = votes(2) + 1;
        else
            votes(4) = votes(4) + 1;
        end
        
        % test class c2 versus c5
        if x * w_2_5 > t_2_5
            votes(2) = votes(2) + 1;
        else
            votes(5) = votes(5) + 1;
        end
        
        % test class c3 versus c4
        if x * w_3_4 > t_3_4
            votes(3) = votes(3) + 1;
        else
            votes(4) = votes(4) + 1;
        end
        
        % test class c3 versus c5
        if x * w_3_5 > t_3_5
            votes(3) = votes(3) + 1;
        else
            votes(5) = votes(5) + 1;
        end
        
        % test class c4 versus c5
        if x * w_4_5 > t_4_5
            votes(4) = votes(4) + 1;
        else
            votes(5) = votes(5) + 1;
        end
        
        % end of voting, tally up
        [m, ind] = max(votes);
        
        if numel(find(votes == m)) > 1
            num_ambig = num_ambig + 1;
            
            % if ambiguous then use strategies to resolve
            if useNone == 1
                ind = 0;
            elseif usePrior == 1
                ind = max(find(votes == m));
            elseif useRunnoff == 1
                
            elseif useTree
                
            end
        end
        
        if ind == p_i
            num_correct = num_correct + 1;
        end
    end

    nambig = num_ambig;
    rate = num_correct / (num_tests  - nambig);
end

function DEBUG_plot_threshold(C1, C2, w, t, titleStr)
    figure;
    hax = axes;
    hold on;
    histfit(C1 * w); hold on; histfit(C2 * w);
    line([t t], get(hax, 'YLim'), 'Color', [1 0 0], 'LineWidth', 3);
    title(titleStr);
end
