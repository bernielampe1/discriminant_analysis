function [rate, nambig] = svm_one2all(data, labels, test_data, test_labels, aheuristic)
    % compute 5 svm models, one-2-all
    modelparams = '-c 1 -q';
    
    % use a tree or prior
    useNone = 0;
    usePrior = 0;
    useMax = 0;
    
    if strcmp(aheuristic, 'none')
        useNone = 1;
    elseif strcmp(aheuristic, 'prior')
        usePrior = 1;
    elseif strcmp(aheuristic, 'max')
        useMax = 1;
    end

    c1_labels = labels;
    c2_labels = labels;
    c3_labels = labels;
    c4_labels = labels;
    c5_labels = labels;
    
    c1_labels(find(labels ~= 1)) = 2; c1_labels(find(labels == 1)) = 1;
    c2_labels(find(labels ~= 2)) = 2; c2_labels(find(labels == 2)) = 1;
    c3_labels(find(labels ~= 3)) = 2; c3_labels(find(labels == 3)) = 1;
    c4_labels(find(labels ~= 4)) = 2; c4_labels(find(labels == 4)) = 1;
    c5_labels(find(labels ~= 5)) = 2; c5_labels(find(labels == 5)) = 1;
    
    model_1 = svmtrain(c1_labels, data, modelparams);
    model_2 = svmtrain(c2_labels, data, modelparams);
    model_3 = svmtrain(c3_labels, data, modelparams);
    model_4 = svmtrain(c4_labels, data, modelparams);
    model_5 = svmtrain(c5_labels, data, modelparams);
    
    % run classifications
    [y1, a1, v1] = svmpredict(test_labels, test_data, model_1, '-q');
    [y2, a2, v2] = svmpredict(test_labels, test_data, model_2, '-q');
    [y3, a3, v3] = svmpredict(test_labels, test_data, model_3, '-q');
    [y4, a4, v4] = svmpredict(test_labels, test_data, model_4, '-q');
    [y5, a5, v5] = svmpredict(test_labels, test_data, model_5, '-q');

    y_all = [y1, y2, y3, y4, y5];
    v_all = [v1, v2, v3, v4, v5];
    nambig = 0;
    num_correct = 0;
    for i = 1:size(y_all, 1)
        y = y_all(i, :);
        v = v_all(i, :);

        % find all indicies that are == 1
        nclasses = numel(find(y == 1));
        
        % check for complete ambiguity
        if nclasses == 0
            nambig = nambig + 1;
        end
        
        % check if unique solution
        if nclasses == 1
            ind = find(y == 1);
        end
        
        % resolve class ambiguity
        if nclasses > 1 || nclasses == 0
            if useNone == 1
                ind = 6;
            elseif usePrior == 1
                ind = max(find(y == 1));
            elseif useMax == 1
                [m, ind] = min(abs(v));
            end
        end
        
        if test_labels(i) == ind
            num_correct = num_correct + 1;
        end
    end
    
    rate = num_correct / (numel(test_labels));
end
