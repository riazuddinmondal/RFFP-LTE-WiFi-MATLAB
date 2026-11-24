function run_RFFP_LSBoost_crosscampaign_fixed(dataFile,outputDir,directions,useParallel)
% Cross-campaign LSBoost, using the original Representation C pipeline.
% Example: run_RFFP_LSBoost_crosscampaign('RFFP_corrected_representations.mat','LSBoost_crosscampaign_results',1:2,true)
% Requires prepareFoldRepresentation.m, materializeAlignedRFFP.m,
% trainLSBoostXY2Workers.m, and positioningMetrics.m on MATLAB path.
if nargin<1 || isempty(dataFile), dataFile='RFFP_corrected_representations.mat'; end
if nargin<2 || isempty(outputDir), outputDir='LSBoost_crosscampaign_results'; end
if nargin<3 || isempty(directions), directions=1:2; end
if nargin<4 || isempty(useParallel), useParallel=false; end
assert(isfile(dataFile),'Missing dataset: %s',dataFile);
for f=["prepareFoldRepresentation","materializeAlignedRFFP","trainLSBoostXY2Workers","positioningMetrics"]
    assert(~isempty(which(char(f))),'Missing required function: %s.m',f);
end
if ~isfolder(outputDir), mkdir(outputDir); end
S=load(dataFile); assert(isfield(S,'reps'),'Expected reps in dataset'); reps=S.reps;
assert(isfield(reps,'targetXY_m'),'Missing targetXY_m');
Y=double(reps.targetXY_m); N=size(Y,1);
assert(N==118930,'Unexpected observation count; verify campaign mapping');
% Metadata is a MATLAB table in the supplied RFFP MAT file.
% isfield(table,'LogID') is false even when the table has that variable.
if isfield(reps,'metadata') && istable(reps.metadata) && ...
        ismember('LogID',reps.metadata.Properties.VariableNames)
    logID=double(reps.metadata.LogID(:));
elseif isfield(reps,'metadata') && isstruct(reps.metadata) && isfield(reps.metadata,'LogID')
    logID=double(reps.metadata.LogID(:));
elseif isfield(S,'parsed') && isfield(S.parsed,'metadata') && ...
        istable(S.parsed.metadata) && ismember('LogID',S.parsed.metadata.Properties.VariableNames)
    logID=double(S.parsed.metadata.LogID(:));
else
    error('LogID not found. Inspect reps.metadata.Properties.VariableNames.');
end
assert(numel(logID)==N && all(isfinite(logID)) && all(ismember(logID,0:37)), ...
    'LogID vector invalid; do not run until corrected');
year=zeros(N,1); year(ismember(logID,22:37))=2014; year(ismember(logID,0:21))=2015;
assert(nnz(year==2014)==32798 && nnz(year==2015)==86132, ...
    'Campaign counts inconsistent with previous verified cross-campaign experiments');
cfg=struct('NumTrees',350,'MaxNumSplits',256,'LearnRate',0.05, ...
    'MinTrainingObservations',50,'MaxLTEFeatures',128,'MaxWLANFeatures',384, ...
    'LTEMissingFloor_dBm',-140,'WLANMissingFloor_dBm',-105,'RandomSeed',42);
if useParallel
    try
        p=gcp('nocreate');
        if isempty(p) || p.NumWorkers~=2
            if ~isempty(p), delete(p); end
            parpool('local',2);
        end
    catch ME
        warning('Parallel pool unavailable: %s. Using serial training.',ME.message);
        useParallel=false;
    end
end
for d=directions(:)'
    assert(ismember(d,[1,2]),'Directions must be 1 and/or 2');
    if d==1, trainYear=2014; testYear=2015; else, trainYear=2015; testYear=2014; end
    trainMask=year==trainYear; testMask=year==testYear;
    assert(~any(trainMask & testMask) && all(trainMask | testMask));
    assert(isempty(intersect(unique(logID(trainMask)),unique(logID(testMask)))),'LogID leakage');
    resultFile=fullfile(outputDir,sprintf('LSBoost_crosscampaign_F%02d_results.mat',d));
    modelFile=fullfile(outputDir,sprintf('LSBoost_crosscampaign_F%02d_models.mat',d));
    if isfile(resultFile) || isfile(modelFile)
        warning('Existing result/model for direction %d; skipping to prevent overwrite.',d); continue;
    end
    fprintf('\nLSBoost cross-campaign %d -> %d | train %d | test %d\n', ...
        trainYear,testYear,nnz(trainMask),nnz(testMask));
    totalClock=tic; prepClock=tic;
    [XTrain,XTest,featureInfo]=prepareFoldRepresentation(reps,"C",trainMask,testMask,cfg);
    preparation_s=toc(prepClock);
    assert(size(XTrain,1)==nnz(trainMask) && size(XTest,1)==nnz(testMask));
    assert(all(isfinite(XTrain(:))) && all(isfinite(XTest(:))),'Nonfinite features');
    YTrain=Y(trainMask,:); YTest=Y(testMask,:);
    ym=mean(YTrain,1); ys=std(YTrain,0,1); ys(ys==0 | ~isfinite(ys))=1;
    Yn=(YTrain-ym)./ys;
    trainClock=tic;
    [mx,my]=trainLSBoostXY2Workers(XTrain,Yn,cfg,logical(useParallel),42+d);
    training_s=toc(trainClock);
    inferClock=tic;
    pred=[predict(mx,XTest)*ys(1)+ym(1),predict(my,XTest)*ys(2)+ym(2)];
    inference_s=toc(inferClock);
    assert(size(pred,1)==nnz(testMask) && all(isfinite(pred(:))),'Incomplete or nonfinite predictions');
    error_m=hypot(pred(:,1)-YTest(:,1),pred(:,2)-YTest(:,2));
    metrics=positioningMetrics(YTest,pred);
    coverage=nnz(all(isfinite(pred),2))/size(pred,1);
    result=struct('Direction',d,'TrainYear',trainYear,'TestYear',testYear, ...
        'NumTrain',nnz(trainMask),'NumTest',nnz(testMask), ...
        'TrainLogIDs',unique(logID(trainMask)),'TestLogIDs',unique(logID(testMask)), ...
        'TestRowIndex',find(testMask),'PredictionXY_m',pred,'TrueXY_m',YTest, ...
        'Error_m',error_m,'Metrics',metrics,'Coverage',coverage, ...
        'Preparation_s',preparation_s,'Training_s',training_s, ...
        'Inference_s',inference_s,'Total_s',toc(totalClock),'Config',cfg);
    save(resultFile,'result','-v7.3');
    save(modelFile,'mx','my','ym','ys','featureInfo','cfg','trainYear','testYear','-v7.3');
    fprintf('Mean %.3f m | P68 %.3f m | P95 %.3f m | coverage %.2f%% | total %.1f s\n', ...
        metrics.Mean_m,metrics.P68_m,metrics.P95_m,100*coverage,result.Total_s);
    fprintf('Saved: %s\nSaved: %s\n',resultFile,modelFile);
    clear mx my XTrain XTest
end
end
