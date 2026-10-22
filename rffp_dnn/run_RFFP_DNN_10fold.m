%% GPU DNN evaluation for corrected route-grouped outdoor RFFP
% This script is intentionally separate from the long-running classical
% evaluation. It uses exactly the same outer fold assignment (seed 42),
% training-only feature construction, and positioning metrics.
clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(scriptDir);
parserDir = fullfile(projectDir,'rffp_parser');
classicalDir = fullfile(projectDir,'rffp_10fold');
addpath(scriptDir,parserDir,classicalDir);

dataFile = fullfile(projectDir,'RFFP_corrected_representations.mat');
outputDir = fullfile(projectDir,'dnn_results');
checkpointFile = fullfile(outputDir,'corrected_route10fold_DNN_checkpoint.mat');
summaryFile = fullfile(outputDir,'corrected_route10fold_DNN_summary.mat');
if ~isfolder(outputDir), mkdir(outputDir); end
assert(isfile(dataFile),'Representation file not found: %s',dataFile);

% Experiment scope. Representation C is the recommended primary DNN input.
representations = "C";       % May later be changed to ["A","B","C"]
foldsToRun = 1:10;           % Use one fold only for an initial pipeline test
numFolds = 10;
randomSeed = 42;

% Feature configuration is identical to the classical corrected run.
cfg = struct();
cfg.MinTrainingObservations = 50;
cfg.MaxLTEFeatures = 128;
cfg.MaxWLANFeatures = 384;
cfg.LTEMissingFloor_dBm = -140;
cfg.WLANMissingFloor_dBm = -105;

% DNN configuration.
cfg.HiddenUnits = [512 256 128 64];
cfg.Dropout = [0.20 0.15 0.10];
cfg.InitialLearnRate = 1e-3;
cfg.LearnRateDropFactor = 0.5;
cfg.LearnRateDropPeriod = 15;
cfg.L2Regularization = 1e-4;
cfg.MiniBatchSize = 512;
cfg.PredictionMiniBatchSize = 2048;
cfg.MaxValidationEpochs = 100;
cfg.ValidationPatience = 12;
cfg.InnerValidationFolds = 5;
cfg.InnerValidationFold = 1;
cfg.MinimumFinalEpochs = 5;
cfg.ExecutionEnvironment = "auto"; % Automatically uses a supported GPU
cfg.Verbose = true;
cfg.RandomSeed = randomSeed;

S = load(dataFile,'reps');
reps = S.reps;
[foldIndex,routeTable] = makeRouteGroupedFolds(reps.metadata,numFolds,randomSeed);
disp(routeTable);
assert(size(reps.targetXY_m,2)==2,'targetXY_m must contain X and Y columns.');
assert(all(isfinite(reps.targetXY_m),'all'),'Target coordinates contain nonfinite values.');

if isfile(checkpointFile)
    C = load(checkpointFile,'results','completedKeys','savedFoldIndex','savedConfig');
    results = C.results;
    completedKeys = C.completedKeys;
    assert(isequal(C.savedFoldIndex,foldIndex),'Checkpoint fold assignment differs.');
    assert(isequaln(C.savedConfig,cfg), ...
        ['DNN checkpoint configuration differs. Rename the old checkpoint ', ...
         'or restore its configuration before resuming.']);
else
    results = struct([]);
    completedKeys = strings(0,1);
    savedFoldIndex = foldIndex;
    savedConfig = cfg;
    save(checkpointFile,'results','completedKeys','savedFoldIndex', ...
        'savedConfig','routeTable','-v7.3');
end

Y = double(reps.targetXY_m);
for rep = representations
    for fold = foldsToRun
        key = sprintf('%s_DNN_F%02d',rep,fold);
        if any(completedKeys==key)
            fprintf('Skipping completed %s\n',key);
            continue;
        end

        outerTrainMask = foldIndex~=fold;
        outerTestMask = foldIndex==fold;
        fprintf('\n%s: grouped inner validation ...\n',key);

        % The validation split contains complete logs only. Feature selection
        % and scaling for this stage are learned without validation routes.
        [innerFitMask,innerValMask,innerRouteTable] = makeRFFPInnerValidation( ...
            reps.metadata,outerTrainMask,cfg.InnerValidationFolds, ...
            cfg.InnerValidationFold,randomSeed+1000+fold);
        [XFit,XVal,innerFeatureInfo] = prepareFoldRepresentation( ...
            reps,rep,innerFitMask,innerValMask,cfg);
        YFit = Y(innerFitMask,:);
        YVal = Y(innerValMask,:);

        rng(randomSeed+fold,'twister');
        validationTimer = tic;
        [bestEpoch,validationInfo] = selectRFFPDNNEpoch( ...
            XFit,YFit,XVal,YVal,cfg);
        validationRuntime = toc(validationTimer);
        clear XFit XVal YFit YVal

        fprintf('%s: selected %d epoch(s); rebuilding outer-fold features ...\n', ...
            key,bestEpoch);
        [XTrain,XTest,featureInfo] = prepareFoldRepresentation( ...
            reps,rep,outerTrainMask,outerTestMask,cfg);
        YTrain = Y(outerTrainMask,:);
        YTest = Y(outerTestMask,:);

        rng(randomSeed+10000+fold,'twister');
        finalTimer = tic;
        [net,targetCenter,targetScale,trainingInfo] = trainFinalRFFPDNN( ...
            XTrain,YTrain,bestEpoch,cfg);
        predNormalized = predict(net,XTest, ...
            'MiniBatchSize',cfg.PredictionMiniBatchSize);
        pred = double(predNormalized).*targetScale + targetCenter;
        finalRuntime = toc(finalTimer);
        totalRuntime = validationRuntime + finalRuntime;

        M = positioningMetrics(YTest,pred);
        predictionFile = fullfile(outputDir,sprintf('%s_predictions.mat',key));
        testRows = find(outerTestMask);
        testLogID = reps.metadata.LogID(outerTestMask);
        save(predictionFile,'pred','YTest','testRows','testLogID','fold','rep', ...
            'targetCenter','targetScale','bestEpoch','-v7.3');

        historyFile = fullfile(outputDir,sprintf('%s_training_history.mat',key));
        save(historyFile,'validationInfo','trainingInfo','innerRouteTable', ...
            'innerFeatureInfo','featureInfo','cfg','-v7.3');

        modelFile = fullfile(outputDir,sprintf('%s_model.mat',key));
        save(modelFile,'net','targetCenter','targetScale','featureInfo', ...
            'bestEpoch','cfg','-v7.3');

        row = struct('Key',key,'Representation',char(rep),'Method','DNN', ...
            'Fold',fold,'NumTrain',nnz(outerTrainMask), ...
            'NumTest',nnz(outerTestMask),'BestEpoch',bestEpoch, ...
            'ValidationRuntime_s',validationRuntime, ...
            'FinalRuntime_s',finalRuntime,'Runtime_s',totalRuntime, ...
            'Metrics',M,'FeatureInfo',featureInfo, ...
            'PredictionFile',predictionFile,'ModelFile',modelFile, ...
            'HistoryFile',historyFile);

        if isempty(results) || isempty(fieldnames(results))
            results = row;
        else
            row = orderfields(row,results);
            results(end+1,1) = row; %#ok<SAGROW>
        end
        completedKeys(end+1,1) = key; %#ok<SAGROW>
        savedFoldIndex = foldIndex;
        savedConfig = cfg;
        save(checkpointFile,'results','completedKeys','savedFoldIndex', ...
            'savedConfig','routeTable','-v7.3');

        fprintf(['%s | Mean %.2f m | Median %.2f m | P68 %.2f m | ', ...
            'P95 %.2f m | epoch %d | %.1f s\n'],key,M.Mean_m,M.Median_m, ...
            M.P68_m,M.P95_m,bestEpoch,totalRuntime);
        clear net XTrain XTest pred predNormalized featureInfo
    end
end

summary = summarizeRFFPDNNResults(results);
disp(summary);
save(summaryFile,'summary','results','routeTable','cfg','-v7.3');

