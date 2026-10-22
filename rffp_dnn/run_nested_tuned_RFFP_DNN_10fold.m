
%% Nested, leakage-free tuning of Representation C DNN
clear; clc;
scriptDir=fileparts(mfilename('fullpath')); projectDir=fileparts(scriptDir);
addpath(scriptDir,fullfile(projectDir,'rffp_parser'), ...
    fullfile(projectDir,'rffp_10fold'));
dataFile=fullfile(projectDir,'RFFP_corrected_representations.mat');
outputDir=fullfile(projectDir,'dnn_tuned_results');
if ~isfolder(outputDir),mkdir(outputDir);end
checkpointFile=fullfile(outputDir,'nested_tuned_DNN_checkpoint.mat');
assert(isfile(dataFile),'Representation file not found: %s',dataFile);
foldsToRun=1:10; numFolds=10; seed=42; rep="C";
base=struct();
base.MinTrainingObservations=50; base.MaxLTEFeatures=128;
base.MaxWLANFeatures=384; base.LTEMissingFloor_dBm=-140;
base.WLANMissingFloor_dBm=-105;
base.HiddenUnits=[512 256 128 64]; base.Dropout=[.20 .15 .10];
base.InitialLearnRate=1e-3; base.LearnRateDropFactor=.5;
base.LearnRateDropPeriod=15; base.L2Regularization=1e-4;
base.MiniBatchSize=512; base.PredictionMiniBatchSize=2048;
base.MaxValidationEpochs=100; base.ValidationPatience=12;
base.MinimumFinalEpochs=5; base.ExecutionEnvironment="auto";
base.TailWeight=.10; base.RandomSeed=seed; base.Verbose=true;
candidates=makeRFFPDNNCandidates(base);

S=load(dataFile,'reps'); reps=S.reps; Y=double(reps.targetXY_m);
[foldIndex,routeTable]=makeRouteGroupedFolds(reps.metadata,numFolds,seed);
if isfile(checkpointFile)
    Q=load(checkpointFile,'results','completedKeys','savedFoldIndex','savedBase');
    results=Q.results; completedKeys=Q.completedKeys;
    assert(isequal(Q.savedFoldIndex,foldIndex)); assert(isequaln(Q.savedBase,base));
else
    results=struct([]);completedKeys=strings(0,1);
end

for fold=foldsToRun
    key=sprintf('C_TunedDNN_F%02d',fold);
    if any(completedKeys==key),fprintf('Skipping %s\n',key);continue;end
    outerTrain=foldIndex~=fold; outerTest=foldIndex==fold;
    [fitMask,valMask,innerRoutes]=makeRFFPInnerValidation( ...
        reps.metadata,outerTrain,5,1,seed+1000+fold);
    [XFit,XVal,innerFeatureInfo]=prepareFoldRepresentation( ...
        reps,rep,fitMask,valMask,base);
    YFit=Y(fitMask,:);YVal=Y(valMask,:);

    tuningFile=fullfile(outputDir,sprintf('%s_tuning.mat',key));
    trial=struct([]);
    if isfile(tuningFile),T=load(tuningFile,'trial');trial=T.trial;end
    for c=1:numel(candidates)
        if numel(trial)>=c && isfield(trial(c),'Complete') && trial(c).Complete
            fprintf('%s: skipping candidate %d/%d\n',key,c,numel(candidates));continue
        end
        cfg=candidates(c); rng(seed+100000*fold+c,'twister'); tic;
        [score,M,epoch,info]=evaluateRFFPDNNCandidate( ...
            XFit,YFit,XVal,YVal,cfg);runtime=toc;
        r=struct('Complete',true,'Index',c,'Name',char(cfg.CandidateName), ...
            'Score',score,'Mean_m',M.Mean_m,'P95_m',M.P95_m, ...
            'RMSE_m',M.RMSE_m,'BestEpoch',epoch,'Runtime_s',runtime, ...
            'Config',cfg,'TrainingInfo',info);
        % MATLAB treats a scalar fieldless struct and a populated struct as dissimilar.
        % Also normalize field order when resuming a partial tuning file created by a previous run.
        if isempty(trial) || isempty(fieldnames(trial))
            trial=r;
        else
            missingInNew=setdiff(fieldnames(trial),fieldnames(r));
            extraInNew=setdiff(fieldnames(r),fieldnames(trial));
            assert(isempty(missingInNew) && isempty(extraInNew), ...
                'RFFP:TrialSchemaMismatch', ...
                ['Existing tuning trial schema differs from the current ', ...
                 'code. Rename the fold tuning MAT file and restart it.']);
            r=orderfields(r,trial);
            trial(c)=r; %#ok<SAGROW>
        end
        save(tuningFile,'trial','innerRoutes','innerFeatureInfo','-v7.3');
        fprintf('%s | %d/%d %-12s | score %.2f | mean %.2f | P95 %.2f | epoch %d\n', ...
            key,c,numel(candidates),cfg.CandidateName,score,M.Mean_m,M.P95_m,epoch);
    end
    [~,best]=min([trial.Score]); selected=trial(best).Config;
    bestEpoch=trial(best).BestEpoch;
    clear XFit XVal YFit YVal

    [XTrain,XTest,featureInfo]=prepareFoldRepresentation( ...
        reps,rep,outerTrain,outerTest,base);
    rng(seed+200000+fold,'twister'); tic;
    [net,targetCenter,targetScale,trainingInfo]=trainFinalRFFPDNN( ...
        XTrain,Y(outerTrain,:),bestEpoch,selected);
    pn=predict(net,XTest,'MiniBatchSize',selected.PredictionMiniBatchSize);
    pred=double(pn).*targetScale+targetCenter; runtime=toc;
    YTest=Y(outerTest,:); M=positioningMetrics(YTest,pred);
    predictionFile=fullfile(outputDir,sprintf('%s_predictions.mat',key));
    testLogID=reps.metadata.LogID(outerTest);testRows=find(outerTest);
    save(predictionFile,'pred','YTest','testLogID','testRows','-v7.3');
    modelFile=fullfile(outputDir,sprintf('%s_model.mat',key));
    save(modelFile,'net','targetCenter','targetScale','selected', ...
        'bestEpoch','featureInfo','trainingInfo','-v7.3');
    row=struct('Key',key,'Representation','C','Method','TunedDNN', ...
        'Fold',fold,'NumTrain',nnz(outerTrain),'NumTest',nnz(outerTest), ...
        'Candidate',char(selected.CandidateName),'BestEpoch',bestEpoch, ...
        'FinalRuntime_s',runtime,'Metrics',M,'PredictionFile',predictionFile, ...
        'ModelFile',modelFile,'TuningFile',tuningFile);
    if isempty(results)||isempty(fieldnames(results)),results=row;
    else,row=orderfields(row,results);results(end+1,1)=row;end %#ok<SAGROW>
    completedKeys(end+1,1)=key; savedFoldIndex=foldIndex;savedBase=base; %#ok<SAGROW>
    save(checkpointFile,'results','completedKeys','savedFoldIndex','savedBase', ...
        'routeTable','-v7.3');
    fprintf('%s FINAL | %s | Mean %.2f | Median %.2f | P68 %.2f | P95 %.2f\n', ...
        key,selected.CandidateName,M.Mean_m,M.Median_m,M.P68_m,M.P95_m);
    clear net XTrain XTest pred pn
end

save(fullfile(outputDir,'nested_tuned_DNN_results.mat'),'results','routeTable','base','-v7.3');
