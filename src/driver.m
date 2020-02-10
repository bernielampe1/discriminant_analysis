function [C, A, N] = driver()
    numcurves = 2;
    N = [10; 20; 30; 40; 50; 70; 90; 100; 200; 300; 400; 500];
    C = zeros(numcurves, numel(N));
    A = zeros(numcurves, numel(N));
    
    for i = 1:numel(N)
       [trainData, trainLabels] = genDelta([0.1, 0.2, 0.3, 0.4, 0.5], N(i), N(i));
       [testData, testLabels] = genDelta([0.1, 0.2, 0.3, 0.4, 0.5], N(i), 50); 
       
       C(1, i) = flda_proj_classify(trainData, trainLabels, testData, testLabels, 'mean', 0)
       C(2, i) = flda_proj_classify(trainData, trainLabels, testData, testLabels, 'gauss', 0)
    end
end
