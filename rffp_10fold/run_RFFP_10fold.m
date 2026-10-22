
%% Corrected route-grouped 10-fold LTE/WLAN RFFP evaluation
clear; clc;

dataFile='RFFP_corrected_representations.mat';
checkpointFile='corrected_route10fold_checkpoint.mat';
numFolds=10; randomSeed=42;
representations=["A","B","C"];
methods=["LSBoost","KNN","AHC","FCM"];
foldsToRun=1:numFolds;

% Start with one representation/method for a quick pipeline check, e.g.:
% representations="C"; methods="LSBoost"; foldsToRun=1;
cfg=struct();
cfg.NumTrees=350; cfg.MaxNumSplits=256; cfg.LearnRate=0.05;
cfg.KNNK=5; cfg.KNNBlockSize=250;
cfg.MinTrainingObservations=50; cfg.MaxLTEFeatures=128; cfg.MaxWLANFeatures=384;
cfg.LTEMissingFloor_dBm=-140; cfg.WLANMissingFloor_dBm=-105;
cfg.NumPrototypes=600; cfg.NumClusters=20; cfg.KMeansMaxIter=50;
cfg.FCMExponent=2; cfg.FCMMaxIter=100; cfg.FCMTolerance=1e-4;
cfg.RandomSeed=randomSeed;


% Parallel execution settings are deliberately kept separate from cfg so
% existing scientific checkpoints remain compatible.
requestedWorkers=2;
useParallelLSBoost=false;
parallelPool=[];
try
    if license('test','Distrib_Computing_Toolbox')
        parallelPool=gcp('nocreate');
        if isempty(parallelPool) || parallelPool.NumWorkers~=requestedWorkers
            if ~isempty(parallelPool)
                fprintf('Replacing existing %d-worker pool with %d workers ...\n', ...
                    parallelPool.NumWorkers,requestedWorkers);
                delete(parallelPool);
            end
            parallelPool=parpool('local',requestedWorkers);
        end
        useParallelLSBoost=~isempty(parallelPool) && ...
            parallelPool.NumWorkers==requestedWorkers;
    else
        warning('Parallel Computing Toolbox not available; using serial LSBoost training.');
    end
catch ME
    warning('Could not start the 2-worker pool (%s). Using serial LSBoost training.',ME.message);
    parallelPool=[];
    useParallelLSBoost=false;
end

if useParallelLSBoost
    fprintf('Parallel LSBoost enabled: %d workers (X/Y models train concurrently).\n',requestedWorkers);
else
    fprintf('Parallel LSBoost disabled: serial fallback active.\n');
end

load(dataFile,'reps');
[foldIndex,routeTable]=makeRouteGroupedFolds(reps.metadata,numFolds,randomSeed);
disp(routeTable);

if isfile(checkpointFile)
    load(checkpointFile,'results','completedKeys','savedFoldIndex','savedConfig');
    assert(isequal(savedFoldIndex,foldIndex),'Checkpoint fold assignment differs.');
    assert(isequaln(savedConfig,cfg),'Checkpoint configuration differs.');
else
    results=struct([]); completedKeys=strings(0,1);
    savedFoldIndex=foldIndex; savedConfig=cfg;
    save(checkpointFile,'results','completedKeys','savedFoldIndex','savedConfig','routeTable','-v7.3');
end

Y=reps.targetXY_m;
for rep=representations
    for fold=foldsToRun
        trainMask=foldIndex~=fold; testMask=foldIndex==fold;
        fprintf('\nRepresentation %s, fold %d/%d: preparing features ...\n',rep,fold,numFolds);
        [XTrain,XTest,featureInfo]=prepareFoldRepresentation(reps,rep,trainMask,testMask,cfg);
        YTrain=Y(trainMask,:); YTest=Y(testMask,:);
        for method=methods
            key=sprintf('%s_%s_F%02d',rep,method,fold);
            if any(completedKeys==key),fprintf('Skipping completed %s\n',key);continue;end
            fprintf('Running %s ...\n',key); rng(randomSeed+fold,'twister'); tic;
            switch method
                case "LSBoost"
                    ym=mean(YTrain,1); ys=std(YTrain,0,1); ys(ys==0)=1;
                    Yn=(YTrain-ym)./ys;
                    [mx,my]=trainLSBoostXY2Workers(XTrain,Yn,cfg,useParallelLSBoost,randomSeed+fold);
                    pred=[predict(mx,XTest)*ys(1)+ym(1),predict(my,XTest)*ys(2)+ym(2)];
                case "KNN"
                    pred=predictIDWKNNBlockwise(XTrain,YTrain,XTest,cfg.KNNK,cfg.KNNBlockSize);
                case {"AHC","FCM"}
                    pred=predictPrototypeClusterIDW(XTrain,YTrain,XTest,method,cfg);
            end
            runtime=toc; M=positioningMetrics(YTest,pred);
            row=struct('Key',key,'Representation',char(rep),'Method',char(method), ...
                'Fold',fold,'NumTrain',nnz(trainMask),'NumTest',nnz(testMask), ...
                'Runtime_s',runtime,'Metrics',M,'FeatureInfo',featureInfo);
            % results(end+1)=row; completedKeys(end+1,1)=key; %#ok<SAGROW>
            % Append result safely
            if isempty(results) || isempty(fieldnames(results))
                results = row;
            else
                row = orderfields(row, results);
                results(end+1,1) = row;
            end

completedKeys(end+1,1) = key; %#ok<SAGROW>
            save(checkpointFile,'results','completedKeys','savedFoldIndex','savedConfig','routeTable','-v7.3');
            fprintf('%s | Mean %.2f m | P68 %.2f m | P95 %.2f m | %.1f s\n', ...
                key,M.Mean_m,M.P68_m,M.P95_m,runtime);
        end
        clear XTrain XTest featureInfo
    end
end

summary=summarizeRFFPResults(results);
disp(summary);
parallelRunInfo=struct('RequestedWorkers',requestedWorkers, ...
    'LSBoostParallelEnabled',useParallelLSBoost);
save('corrected_route10fold_summary.mat','summary','routeTable','cfg','parallelRunInfo','-v7.3');
