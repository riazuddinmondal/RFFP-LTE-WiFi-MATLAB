function results = run_RFFP_Hierarchical_FastKAN_V2(dataFile, foldNumber, varargin)
%RUN_RFFP_HIERARCHICAL_FASTKAN_V2 Validation-controlled hierarchical FastKAN positioning.
%
% Designed for RFFP_corrected_representations.mat created by the corrected
% RFFP pipeline. Uses Representation C (identifier-aligned sparse RSS),
% route-grouped outer folds, a global FastKAN and LTE-conditioned local
% FastKAN experts, followed by RSRP-weighted expert fusion.
%
% REQUIREMENTS
%   MATLAB R2022b or newer recommended
%   Deep Learning Toolbox
%   Parallel Computing Toolbox only if you want NVIDIA GPU acceleration
%
% EXAMPLE
%   results = run_RFFP_Hierarchical_FastKAN_V2( ...
%       'RFFP_corrected_representations.mat', 1, ...
%       'FoldFile','corrected_route10fold_checkpoint.mat', ...
%       'Epochs',100,'BatchSize',1024,'ExecutionEnvironment','auto');
%
% QUICK TEST
%   results = run_RFFP_Hierarchical_FastKAN_V2( ...
%       'RFFP_corrected_representations.mat',1, ...
%       'FoldFile','corrected_route10fold_checkpoint.mat', ...
%       'Epochs',10,'MaxExperts',5);
%
% The method deliberately selects all radio features using TRAINING samples
% only, to avoid test-fold leakage.

p = inputParser;
p.addRequired('dataFile',@(x)ischar(x)||isstring(x));
p.addRequired('foldNumber',@(x)isnumeric(x)&&isscalar(x)&&x>=1);
p.addParameter('FoldFile','corrected_route10fold_checkpoint.mat',@(x)ischar(x)||isstring(x));
p.addParameter('Epochs',100,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
p.addParameter('BatchSize',1024,@(x)isnumeric(x)&&isscalar(x)&&x>=16);
p.addParameter('LearningRate',2e-3,@(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('WeightDecay',2e-5,@(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('Patience',12,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
p.addParameter('NumGrids',12,@(x)isnumeric(x)&&isscalar(x)&&x>=3);
p.addParameter('Dropout',0.04,@(x)isnumeric(x)&&isscalar(x)&&x>=0&&x<1);
p.addParameter('RSSJitter_dB',1.2,@(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('CandidateFeatures',768,@(x)isnumeric(x)&&isscalar(x)&&x>=16);
p.addParameter('GlobalFeatures',256,@(x)isnumeric(x)&&isscalar(x)&&x>=8);
p.addParameter('LocalFeatures',160,@(x)isnumeric(x)&&isscalar(x)&&x>=8);
p.addParameter('MinExpertSamples',1500,@(x)isnumeric(x)&&isscalar(x)&&x>=100);
p.addParameter('MaxExperts',15,@(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('FusionStep',0.05,@(x)isnumeric(x)&&isscalar(x)&&x>0&&x<=0.5);
p.addParameter('MinFusionImprovement',0.002,@(x)isnumeric(x)&&isscalar(x)&&x>=0&&x<1);
p.addParameter('FusionObjective','hybrid',@(x)any(strcmpi(x,{'mean','p68','hybrid'})));
p.addParameter('MinExpertValSamples',100,@(x)isnumeric(x)&&isscalar(x)&&x>=20);
p.addParameter('Seed',17,@(x)isnumeric(x)&&isscalar(x));
p.addParameter('ExecutionEnvironment','auto',@(x)any(strcmpi(x,{'auto','cpu','gpu'})));
p.addParameter('OutputDir','KAN_MATLAB_results',@(x)ischar(x)||isstring(x));
p.parse(dataFile,foldNumber,varargin{:});
cfg = p.Results;

% Runtime instrumentation. AlgorithmTotal_s excludes MAT-file save time; the
% 10-fold wrapper separately records full wall-clock time including saves.
tAlgorithmTotal = tic;
tPreprocess = tic;

rng(cfg.Seed,'twister');
if ~isfile(dataFile)
    error('Data file not found: %s',dataFile);
end
if ~isfile(cfg.FoldFile)
    error(['Route-grouped fold file not found: %s\n' ...
        'For publishable comparison, use the same corrected_route10fold_checkpoint.mat ' ...
        'used by the other RFFP models.'],cfg.FoldFile);
end
if ~isfolder(cfg.OutputDir), mkdir(cfg.OutputDir); end

fprintf('\n============================================================\n');
fprintf('Hierarchical LTE/Wi-Fi FastKAN RFFP V2 - MATLAB\n');
fprintf('Outer fold: %d\n',foldNumber);
fprintf('============================================================\n');

%% Load corrected representation C and supporting arrays
fprintf('Loading %s ...\n',dataFile);
S = load(dataFile,'reps','parsed');
if ~isfield(S,'reps') || ~isfield(S.reps,'C_alignedSparse')
    error('Expected reps.C_alignedSparse in %s.',dataFile);
end

X = S.reps.C_alignedSparse.PowerObserved_dBm;
xy = single(S.reps.targetXY_m);
txid = double(S.reps.C_alignedSparse.TransmitterID(:));
wlanStart = double(S.parsed.WLANIDStart);
lteIDs = double(S.reps.B_topK.LTE_IDs);
lteRSRP = single(S.reps.B_topK.LTE_RSRP_dBm);
numPairs = single(S.reps.A_rankPairs.NumPairs(:));

N = size(X,1);
if size(xy,1) ~= N && size(xy,2) == N, xy = xy.'; end
if size(lteIDs,1) ~= N && size(lteIDs,2) == N, lteIDs = lteIDs.'; end
if size(lteRSRP,1) ~= N && size(lteRSRP,2) == N, lteRSRP = lteRSRP.'; end
if numel(numPairs) ~= N, error('NumPairs length does not match N.'); end

isLTE = txid < wlanStart;
fprintf('N=%d, radio columns=%d (LTE=%d, Wi-Fi=%d)\n', ...
    N,size(X,2),nnz(isLTE),nnz(~isLTE));

%% Exact route-grouped outer fold
foldVec = loadFoldVectorRobust(cfg.FoldFile,N);
if ~ismember(foldNumber,unique(foldVec))
    error('Fold %d does not exist in the loaded fold vector.',foldNumber);
end
idxTest = find(foldVec==foldNumber);
idxOuterTrain = find(foldVec~=foldNumber);
[idxTrain,idxVal] = splitTrainVal(idxOuterTrain,xy,cfg.Seed,0.10);
fprintf('Split: train=%d | val=%d | test=%d\n',numel(idxTrain),numel(idxVal),numel(idxTest));

%% Execution environment
useGPU = false;
if strcmpi(cfg.ExecutionEnvironment,'gpu')
    if ~canUseGPU, error('ExecutionEnvironment=gpu requested but no supported GPU is available.'); end
    useGPU = true;
elseif strcmpi(cfg.ExecutionEnvironment,'auto')
    useGPU = canUseGPU;
end
if useGPU
    g = gpuDevice;
    fprintf('Execution: GPU (%s)\n',g.Name);
else
    fprintf('Execution: CPU\n');
end

%% Training-only candidate radio pool, LTE/Wi-Fi balanced
nLTEcand = min(96,floor(cfg.CandidateFeatures/4));
nWiFicand = cfg.CandidateFeatures - nLTEcand;
candidateIdx = chooseCandidates(X,idxTrain,isLTE,nLTEcand,nWiFicand);

globalFeat = chooseFromCandidates(X,idxTrain,candidateIdx,cfg.GlobalFeatures);
fprintf('Global FastKAN selected %d radio features (+ radio-count feature).\n',numel(globalFeat));

Xtr = denseFeatures(X,idxTrain,globalFeat,numPairs);
Xva = denseFeatures(X,idxVal,globalFeat,numPairs);
Xte = denseFeatures(X,idxTest,globalFeat,numPairs);

%% Global FastKAN
preprocessing_s = toc(tPreprocess);
fprintf('\nTraining GLOBAL FastKAN ...\n');
widthsG = [size(Xtr,2) 96 48 2];
tGlobalTrain = tic;
[globalNet,gCenter,gScale,gHistory] = trainFastKAN( ...
    Xtr,xy(idxTrain,:),Xva,xy(idxVal,:),widthsG,cfg,useGPU,cfg.Seed);
globalTraining_s = toc(tGlobalTrain);

tGlobalInfer = tic;
pGlobal = predictFastKAN(globalNet,Xte,gCenter,gScale,useGPU,4096);
globalInference_s = toc(tGlobalInfer);
globalFastKAN_s = globalTraining_s + globalInference_s;
globalMetrics = positionMetrics(xy(idxTest,:),pGlobal);
printMetrics('Global FastKAN',globalMetrics);

clear Xtr Xva Xte

%% LTE-conditioned local experts with common validation set
tLocalStage = tic;
localTraining_s = 0;
% V2 change: every local expert trains ONLY on idxTrain and is judged ONLY
% on the untouched common idxVal. This makes expert acceptance and fusion
% weights directly comparable with the global network.
trainLTE = lteIDs(idxTrain,:);
validLTE = trainLTE(trainLTE>0 & isfinite(trainLTE));
[uLTE,~,ic] = unique(validLTE);
counts = accumarray(ic,1);
[countsSorted,ord] = sort(counts,'descend');
uLTE = uLTE(ord);
expertIDs = uLTE(countsSorted >= cfg.MinExpertSamples);
expertIDs = expertIDs(1:min(cfg.MaxExperts,numel(expertIDs)));

fprintf('\nTraining up to %d LTE-conditioned local FastKAN experts ...\n',numel(expertIDs));
experts = cell(numel(expertIDs),1);
accepted = false(numel(expertIDs),1);

% Global predictions on the untouched validation set are reused when each
% expert is judged and later when the overall meta-fusion is calibrated.
XvaGlobal = denseFeatures(X,idxVal,globalFeat,numPairs);
tGlobalValInfer = tic;
pGlobalVal = predictFastKAN(globalNet,XvaGlobal,gCenter,gScale,useGPU,4096);
globalValidationInference_s = toc(tGlobalValInfer);
clear XvaGlobal

for ei = 1:numel(expertIDs)
    tExpertTotal = tic;
    lid = expertIDs(ei);
    etr = idxTrain(any(lteIDs(idxTrain,:)==lid,2));
    eva = idxVal(any(lteIDs(idxVal,:)==lid,2));

    if numel(etr) < cfg.MinExpertSamples || numel(eva) < cfg.MinExpertValSamples
        fprintf('  Expert %02d/%02d | LTE %g skipped (train=%d, val=%d)\n', ...
            ei,numel(expertIDs),lid,numel(etr),numel(eva));
        continue;
    end

    feat = chooseFromCandidates(X,etr,candidateIdx,cfg.LocalFeatures);
    A = denseFeatures(X,etr,feat,numPairs);
    B = denseFeatures(X,eva,feat,numPairs);

    fprintf('  Expert %02d/%02d | LTE %g | train=%d | val=%d | features=%d\n', ...
        ei,numel(expertIDs),lid,numel(etr),numel(eva),numel(feat));
    widthsL = [size(A,2) 64 32 2];
    tExpertTrain = tic;
    [net,c,scl,hist] = trainFastKAN(A,xy(etr,:),B,xy(eva,:), ...
        widthsL,cfg,useGPU,cfg.Seed+ei);
    expertTraining_s = toc(tExpertTrain);
    localTraining_s = localTraining_s + expertTraining_s;

    pLocalVal = predictFastKAN(net,B,c,scl,useGPU,4096);
    [tfv,locv] = ismember(eva,idxVal);
    if ~all(tfv), error('Internal validation-index mapping failure.'); end
    pG = pGlobalVal(locv,:);
    yV = xy(eva,:);

    [bestAlpha,bestScore,baseScore,bestMetrics] = optimizeBlend( ...
        yV,pG,pLocalVal,cfg.FusionStep,cfg.FusionObjective);
    improvement = (baseScore-bestScore)/max(baseScore,eps);
    useExpert = bestAlpha > 0 && improvement >= cfg.MinFusionImprovement;

    mg = positionMetrics(yV,pG);
    ml = positionMetrics(yV,pLocalVal);
    fprintf(['      validation: Global Mean %.2f/P68 %.2f | Local Mean %.2f/P68 %.2f | ' ...
        'alpha=%.2f | objective improvement %.2f%% -> %s\n'], ...
        mg.MeanPE_m,mg.P68_m,ml.MeanPE_m,ml.P68_m,bestAlpha,100*improvement, ...
        ternary(useExpert,'ACCEPT','REJECT'));

    expertTotal_s = toc(tExpertTotal);
    experts{ei} = struct('LTE_ID',lid,'Net',net,'Center',c,'Scale',scl, ...
        'Features',feat,'NTrain',numel(etr),'NVal',numel(eva),'History',hist, ...
        'Alpha',bestAlpha,'UseForFusion',useExpert,'ValidationGlobalMetrics',mg, ...
        'ValidationLocalMetrics',ml,'ValidationBlendedMetrics',bestMetrics, ...
        'ValidationObjectiveImprovement',improvement, ...
        'TrainingTime_s',expertTraining_s,'TotalExpertTime_s',expertTotal_s);
    fprintf('      runtime: training %.1f min | expert total %.1f min\n', ...
        expertTraining_s/60,expertTotal_s/60);
    accepted(ei) = useExpert;
    clear A B net hist pLocalVal
end

localExpertsStage_s = toc(tLocalStage);
fprintf('Accepted %d/%d trained candidate experts for fusion.\n',nnz(accepted),numel(expertIDs));

%% Validation-controlled ensemble and global safety fusion
tFusionCalibration = tic;
% Build the accepted-expert ensemble on the COMMON validation set. Each
% expert first receives its own conservative global/local alpha, then the
% whole ensemble receives a second meta-alpha against the global FastKAN.
[pExpertVal,usedExpertVal] = predictExpertEnsemble( ...
    X,idxVal,numPairs,lteIDs,lteRSRP,experts,pGlobalVal,useGPU);

if any(usedExpertVal)
    [metaAlpha,metaScore,globalScore,~] = optimizeBlend( ...
        xy(idxVal(usedExpertVal),:),pGlobalVal(usedExpertVal,:), ...
        pExpertVal(usedExpertVal,:),cfg.FusionStep,cfg.FusionObjective);
    metaImprovement = (globalScore-metaScore)/max(globalScore,eps);
    if metaImprovement < cfg.MinFusionImprovement
        metaAlpha = 0;
    end
else
    metaAlpha = 0;
    metaImprovement = 0;
end
fprintf('Meta-fusion validation alpha = %.2f (objective improvement %.2f%%).\n', ...
    metaAlpha,100*metaImprovement);

pValFinal = pGlobalVal;
if metaAlpha > 0
    pValFinal(usedExpertVal,:) = single((1-metaAlpha)*double(pGlobalVal(usedExpertVal,:)) + ...
        metaAlpha*double(pExpertVal(usedExpertVal,:)));
end
validationGlobalMetrics = positionMetrics(xy(idxVal,:),pGlobalVal);
validationFinalMetrics = positionMetrics(xy(idxVal,:),pValFinal);
printMetrics('Validation Global FastKAN',validationGlobalMetrics);
printMetrics('Validation V2 fused FastKAN',validationFinalMetrics);
fusionCalibration_s = toc(tFusionCalibration);

%% Test prediction using ONLY validation-calibrated fusion parameters
tHierarchicalTest = tic;
[pExpertTest,usedLocal] = predictExpertEnsemble( ...
    X,idxTest,numPairs,lteIDs,lteRSRP,experts,pGlobal,useGPU);

pFinal = pGlobal;
if metaAlpha > 0
    pFinal(usedLocal,:) = single((1-metaAlpha)*double(pGlobal(usedLocal,:)) + ...
        metaAlpha*double(pExpertTest(usedLocal,:)));
end

hierMetrics = positionMetrics(xy(idxTest,:),pFinal);
localCoverage = mean(usedLocal);
hierarchicalTestInference_s = toc(tHierarchicalTest);
hierarchicalStage_s = localExpertsStage_s + fusionCalibration_s + hierarchicalTestInference_s;
algorithmTotal_s = toc(tAlgorithmTotal);
printMetrics('Hierarchical FastKAN V2',hierMetrics);
fprintf('Accepted-expert test coverage: %.2f %%\n',100*localCoverage);

fprintf('\nRuntime breakdown for Fold %d:\n',foldNumber);
fprintf('  Preprocessing/data preparation : %s\n',formatDuration(preprocessing_s));
fprintf('  Global FastKAN training        : %s\n',formatDuration(globalTraining_s));
fprintf('  Global FastKAN test inference  : %s\n',formatDuration(globalInference_s));
fprintf('  Global FastKAN total           : %s\n',formatDuration(globalFastKAN_s));
fprintf('  Local FastKAN training only    : %s\n',formatDuration(localTraining_s));
fprintf('  Local-expert stage total       : %s\n',formatDuration(localExpertsStage_s));
fprintf('  Fusion calibration             : %s\n',formatDuration(fusionCalibration_s));
fprintf('  Hierarchical test inference    : %s\n',formatDuration(hierarchicalTestInference_s));
fprintf('  Hierarchical stage total       : %s\n',formatDuration(hierarchicalStage_s));
fprintf('  TOTAL KAN algorithm time       : %s\n',formatDuration(algorithmTotal_s));

%% Save outputs
err = sqrt(sum((double(pFinal)-double(xy(idxTest,:))).^2,2));
results = struct;
results.Method = 'Hierarchical LTE-WiFi FastKAN V2';
results.Fold = foldNumber;
results.Config = cfg;
results.GlobalMetrics = globalMetrics;
results.ValidationGlobalMetrics = validationGlobalMetrics;
results.ValidationFusedMetrics = validationFinalMetrics;
results.MetaFusionAlpha = metaAlpha;
results.MetaFusionImprovement = metaImprovement;
results.HierarchicalMetrics = hierMetrics;
results.LocalCoverage = localCoverage;
results.TestIndex = idxTest;
results.TrueXY_m = xy(idxTest,:);
results.PredictedXY_m = pFinal;
results.GlobalPredictedXY_m = pGlobal;
results.PositionError_m = err;
results.UsedLocalExpert = usedLocal;
results.GlobalFeatures = globalFeat;
results.CandidateFeatures = candidateIdx;
results.ExpertIDs = expertIDs;
results.AcceptedExpertMask = accepted;
results.ExpertSummary = summarizeExperts(experts);
results.GlobalTrainingHistory = gHistory;
results.Timing = struct( ...
    'Preprocessing_s',preprocessing_s, ...
    'GlobalTraining_s',globalTraining_s, ...
    'GlobalInference_s',globalInference_s, ...
    'GlobalFastKAN_s',globalFastKAN_s, ...
    'GlobalValidationInference_s',globalValidationInference_s, ...
    'LocalFastKANTraining_s',localTraining_s, ...
    'LocalExpertsStage_s',localExpertsStage_s, ...
    'FusionCalibration_s',fusionCalibration_s, ...
    'HierarchicalInference_s',hierarchicalTestInference_s, ...
    'HierarchicalStage_s',hierarchicalStage_s, ...
    'AlgorithmTotal_s',algorithmTotal_s);

modelFile = fullfile(cfg.OutputDir,sprintf('FastKAN_V2_fold%d_models.mat',foldNumber));
resultFile = fullfile(cfg.OutputDir,sprintf('FastKAN_V2_fold%d_results.mat',foldNumber));
save(resultFile,'results','-v7.3');
save(modelFile,'globalNet','gCenter','gScale','globalFeat','experts','cfg','-v7.3');

fprintf('\nSaved:\n  %s\n  %s\n',resultFile,modelFile);
end

%% ========================================================================
function net = buildFastKAN(widths,numGrids,dropout)
layers = featureInputLayer(widths(1),'Normalization','none','Name','input');
for k = 1:numel(widths)-1
    layers = [layers
        FastKANLayer(widths(k),widths(k+1),numGrids,sprintf('kan%d',k))]; %#ok<AGROW>
    if k < numel(widths)-1 && dropout > 0
        layers = [layers
            dropoutLayer(dropout,'Name',sprintf('drop%d',k))]; %#ok<AGROW>
    end
end
net = dlnetwork(layerGraph(layers));
end

function [net,center,scale,history] = trainFastKAN(Xtr,Ytr,Xva,Yva,widths,cfg,useGPU,seed)
rng(seed,'twister');
center = mean(Ytr,1,'omitnan');
sd = std(Ytr,0,1,'omitnan');
scale = max(mean(sd),50);
YtrN = single((double(Ytr)-double(center))/double(scale));
YvaN = single((double(Yva)-double(center))/double(scale));

net = buildFastKAN(widths,cfg.NumGrids,cfg.Dropout);
trAvg = [];
trAvgSq = [];
iteration = 0;
lr = cfg.LearningRate;
bestLoss = inf;
bestNet = net;
stale = 0;
plateau = 0;

history.Epoch = zeros(cfg.Epochs,1);
history.TrainLoss = nan(cfg.Epochs,1);
history.ValLoss = nan(cfg.Epochs,1);
history.LearningRate = nan(cfg.Epochs,1);

n = size(Xtr,1);
for epoch = 1:cfg.Epochs
    order = randperm(n);
    running = 0;
    seen = 0;

    for s = 1:cfg.BatchSize:n
        iteration = iteration + 1;
        ids = order(s:min(s+cfg.BatchSize-1,n));
        xb = single(Xtr(ids,:));
        yb = single(YtrN(ids,:));
        if cfg.RSSJitter_dB > 0
            xb = xb + single(randn(size(xb))*(cfg.RSSJitter_dB*2/105));
        end
        dlX = dlarray(xb.','CB');
        dlY = dlarray(yb.','CB');
        if useGPU
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
        end

        [loss,grads] = dlfeval(@modelLoss,net,dlX,dlY,cfg.WeightDecay);
        grads = clipGradients(grads,5.0);
        [net,trAvg,trAvgSq] = adamupdate(net,grads,trAvg,trAvgSq, ...
            iteration,lr,0.9,0.999,1e-8);

        bs = numel(ids);
        running = running + double(gather(extractdata(loss)))*bs;
        seen = seen + bs;
    end

    valLoss = evaluateLoss(net,Xva,YvaN,cfg.BatchSize,useGPU);
    trainLoss = running/max(seen,1);
    history.Epoch(epoch) = epoch;
    history.TrainLoss(epoch) = trainLoss;
    history.ValLoss(epoch) = valLoss;
    history.LearningRate(epoch) = lr;

    if epoch==1 || mod(epoch,10)==0
        fprintf('    epoch %3d: train=%.5f val=%.5f lr=%.3g\n', ...
            epoch,trainLoss,valLoss,lr);
    end

    if valLoss < bestLoss - 1e-5
        bestLoss = valLoss;
        bestNet = net;
        stale = 0;
        plateau = 0;
    else
        stale = stale + 1;
        plateau = plateau + 1;
    end

    if plateau >= 5
        lr = max(lr*0.5,1e-5);
        plateau = 0;
    end
    if stale >= cfg.Patience
        fprintf('    early stop at epoch %d (best val %.5f)\n',epoch,bestLoss);
        break;
    end
end
net = bestNet;
keep = history.Epoch>0;
history.Epoch = history.Epoch(keep);
history.TrainLoss = history.TrainLoss(keep);
history.ValLoss = history.ValLoss(keep);
history.LearningRate = history.LearningRate(keep);
end

function [loss,grads] = modelLoss(net,dlX,dlY,weightDecay)
dlP = forward(net,dlX);
e = dlP-dlY;
beta = 0.15;
a = abs(e);
q = min(a,beta);
loss = mean(0.5*(q.^2)/beta + (a-q),'all');

if weightDecay > 0
    reg = dlarray(single(0));
    for i = 1:size(net.Learnables,1)
        nm = net.Learnables.Parameter(i);
        if contains(nm,'Weights','IgnoreCase',true)
            w = net.Learnables.Value{i};
            reg = reg + sum(w.^2,'all');
        end
    end
    loss = loss + weightDecay*reg;
end
grads = dlgradient(loss,net.Learnables);
end

function grads = clipGradients(grads,maxNorm)
ss = 0;
for i=1:size(grads,1)
    g = grads.Value{i};
    if ~isempty(g)
        gd = gather(extractdata(g));
        ss = ss + sum(double(gd(:)).^2);
    end
end
gn = sqrt(ss);
if gn > maxNorm
    f = maxNorm/(gn+eps);
    for i=1:size(grads,1)
        if ~isempty(grads.Value{i})
            grads.Value{i} = grads.Value{i}*f;
        end
    end
end
end

function v = evaluateLoss(net,X,Y,batch,useGPU)
n=size(X,1); total=0; seen=0;
for s=1:batch:n
    ids=s:min(s+batch-1,n);
    dlX=dlarray(single(X(ids,:).'),'CB');
    if useGPU, dlX=gpuArray(dlX); end
    p=predict(net,dlX);
    yt=single(Y(ids,:).');
    if useGPU, yt=gpuArray(yt); end
    e=p-dlarray(yt,'CB');
    beta=.15; a=abs(e); q=min(a,beta);
    L=mean(0.5*(q.^2)/beta+(a-q),'all');
    bs=numel(ids); total=total+double(gather(extractdata(L)))*bs; seen=seen+bs;
end
v=total/max(seen,1);
end

function P = predictFastKAN(net,X,center,scale,useGPU,batch)
n=size(X,1); P=zeros(n,2,'single');
for s=1:batch:n
    ids=s:min(s+batch-1,n);
    dlX=dlarray(single(X(ids,:).'),'CB');
    if useGPU, dlX=gpuArray(dlX); end
    z=predict(net,dlX);
    z=gather(extractdata(z)).';
    P(ids,:)=single(double(z)*double(scale)+double(center));
end
end

function feat = chooseCandidates(X,rows,isLTE,nLTE,nWiFi)
cnt = full(sum(X(rows,:)~=0,1));
lte = find(isLTE); wifi=find(~isLTE);
[~,o1]=sort(cnt(lte),'descend');
[~,o2]=sort(cnt(wifi),'descend');
lte=lte(o1(1:min(nLTE,numel(o1))));
wifi=wifi(o2(1:min(nWiFi,numel(o2))));
feat=[lte(:);wifi(:)];
end

function feat = chooseFromCandidates(X,rows,candidates,nfeat)
cnt=full(sum(X(rows,candidates)~=0,1));
[~,ord]=sort(cnt,'descend');
ord=ord(1:min(nfeat,numel(ord)));
feat=candidates(ord);
end

function A = denseFeatures(X,rows,feat,numPairs)
A=single(full(X(rows,feat)));
miss=(A==0);
A=min(max(A,-140),-35);
A=(A+87.5)/52.5;
A(miss)=-1;
c=single(min(max(numPairs(rows),0),94))/47-1;
A=[A c];
end

function [tr,va] = splitTrainVal(idx,xy,seed,frac)
% Deterministic spatially ordered holdout, analogous to the Python version.
[~,ord]=sortrows(double(xy(idx,:)),[2 1]);
ordered=idx(ord);
stride=max(round(1/frac),2);
offset=mod(seed-1,stride)+1;
va=ordered(offset:stride:end);
isVal=false(max(idx),1); isVal(va)=true;
tr=idx(~isVal(idx));
end

function T = summarizeExperts(experts)
rows = cell(numel(experts),1);
for i=1:numel(experts)
    E=experts{i};
    if isempty(E), continue; end
    rows{i}=struct('LTE_ID',E.LTE_ID,'NTrain',E.NTrain,'NVal',E.NVal, ...
        'Alpha',E.Alpha,'UseForFusion',E.UseForFusion, ...
        'ValidationObjectiveImprovement',E.ValidationObjectiveImprovement, ...
        'TrainingTime_s',E.TrainingTime_s,'TotalExpertTime_s',E.TotalExpertTime_s, ...
        'GlobalMean_m',E.ValidationGlobalMetrics.MeanPE_m, ...
        'LocalMean_m',E.ValidationLocalMetrics.MeanPE_m, ...
        'BlendedMean_m',E.ValidationBlendedMetrics.MeanPE_m, ...
        'GlobalP68_m',E.ValidationGlobalMetrics.P68_m, ...
        'LocalP68_m',E.ValidationLocalMetrics.P68_m, ...
        'BlendedP68_m',E.ValidationBlendedMetrics.P68_m, ...
        'GlobalP95_m',E.ValidationGlobalMetrics.P95_m, ...
        'LocalP95_m',E.ValidationLocalMetrics.P95_m, ...
        'BlendedP95_m',E.ValidationBlendedMetrics.P95_m);
end
rows=rows(~cellfun(@isempty,rows));
if isempty(rows)
    T=table();
else
    T=struct2table(vertcat(rows{:}));
end
end

function [P,used] = predictExpertEnsemble(X,rows,numPairs,lteIDs,lteRSRP,experts,pGlobal,useGPU)
% RSRP-weighted ensemble of VALIDATED experts. Each expert prediction is
% first blended with the global prediction using its validation-optimized
% Alpha. Rejected experts never contribute.
n = numel(rows);
accum = zeros(n,2,'double');
wsum = zeros(n,1,'double');

for ei = 1:numel(experts)
    E = experts{ei};
    if isempty(E) || ~E.UseForFusion || E.Alpha <= 0, continue; end
    lid = E.LTE_ID;
    [r,c] = find(lteIDs(rows,:)==lid);
    if isempty(r), continue; end
    targetRows = unique(r);

    rr = -140*ones(numel(targetRows),1,'single');
    valsAll = lteRSRP(rows,:);
    vals = valsAll(sub2ind(size(valsAll),r,c));
    [~,pos] = ismember(r,targetRows);
    for k = 1:numel(r)
        if isfinite(vals(k))
            rr(pos(k)) = max(rr(pos(k)),vals(k));
        end
    end

    XX = denseFeatures(X,rows(targetRows),E.Features,numPairs);
    pL = predictFastKAN(E.Net,XX,E.Center,E.Scale,useGPU,4096);
    pAdj = (1-E.Alpha)*double(pGlobal(targetRows,:)) + E.Alpha*double(pL);

    % Stronger LTE signals receive more weight, while exponent clipping
    % prevents one radio from numerically dominating the ensemble.
    ww = exp((double(min(max(rr,-140),-40))+80)/10);
    ww = min(max(ww,0.02),50);
    accum(targetRows,:) = accum(targetRows,:) + pAdj.*ww;
    wsum(targetRows) = wsum(targetRows) + ww;
end

used = wsum>0;
P = pGlobal;
P(used,:) = single(accum(used,:)./wsum(used));
end

function [bestAlpha,bestScore,baseScore,bestMetrics] = optimizeBlend(Y,pG,pL,step,objective)
alphas = 0:step:1;
bestAlpha = 0;
bestMetrics = positionMetrics(Y,pG);
baseScore = metricObjective(bestMetrics,objective);
bestScore = baseScore;
for a = alphas(2:end)
    P = single((1-a)*double(pG) + a*double(pL));
    M = positionMetrics(Y,P);
    sc = metricObjective(M,objective);
    if sc < bestScore - 1e-12
        bestScore = sc;
        bestAlpha = a;
        bestMetrics = M;
    end
end
end

function s = metricObjective(M,objective)
switch lower(objective)
    case 'mean'
        s = M.MeanPE_m;
    case 'p68'
        s = M.P68_m;
    otherwise
        % Balanced objective: prioritize central accuracy but retain a
        % meaningful penalty for the long-error tail.
        s = 0.55*M.MeanPE_m + 0.30*M.P68_m + 0.15*M.P95_m;
end
end


function txt = formatDuration(secondsValue)
secondsValue = double(secondsValue);
if ~isfinite(secondsValue)
    txt = 'NaN';
    return;
end
h = floor(secondsValue/3600);
m = floor(mod(secondsValue,3600)/60);
s = mod(secondsValue,60);
if h > 0
    txt = sprintf('%dh %02dm %05.2fs',h,m,s);
elseif m > 0
    txt = sprintf('%dm %05.2fs',m,s);
else
    txt = sprintf('%.2fs',s);
end
end

function out = ternary(cond,a,b)
if cond, out=a; else, out=b; end
end

function fold = loadFoldVectorRobust(file,N)
D=load(file);
[cands,names]=collectNumericVectors(D,'');
bestScore=-inf; best=[]; bestName='';
for i=1:numel(cands)
    a=cands{i};
    if ~isvector(a) || numel(a)~=N || any(~isfinite(double(a(:)))), continue; end
    ad=double(a(:)); ai=round(ad); u=unique(ai);
    if max(abs(ad-ai))>1e-9 || numel(u)<2 || numel(u)>30 || min(u)<0 || max(u)>30
        continue;
    end
    score=0; nm=lower(names{i});
    if contains(nm,'fold')||contains(nm,'cv')||contains(nm,'group'), score=score+10; end
    if min(u)==1, score=score+5; end
    if score>bestScore
        bestScore=score; best=ai; bestName=names{i};
    end
end
if isempty(best)
    error('Could not locate an N=%d fold vector in %s.',N,file);
end
if min(best)==0, best=best+1; end
fold=best;
fprintf('Loaded fold vector "%s" with folds: %s\n',bestName,mat2str(unique(fold).'));
end

function [vals,names] = collectNumericVectors(S,prefix)
vals={}; names={};
f=fieldnames(S);
for i=1:numel(f)
    name=f{i}; v=S.(name); path=[prefix '/' name];
    if isstruct(v) && isscalar(v)
        [vv,nn]=collectNumericVectors(v,path); vals=[vals vv]; names=[names nn]; %#ok<AGROW>
    elseif isnumeric(v) || islogical(v)
        vals{end+1}=v; names{end+1}=path; %#ok<AGROW>
    end
end
end

function M = positionMetrics(Y,P)
e=sqrt(sum((double(P)-double(Y)).^2,2));
M.N=numel(e);
M.MeanPE_m=mean(e);
M.MedianPE_m=median(e);
M.RMSE_m=sqrt(mean(e.^2));
M.P68_m=percentileLocal(e,68);
M.P95_m=percentileLocal(e,95);
M.MaxPE_m=max(e);
end

function q=percentileLocal(x,p)
x=sort(double(x(:))); n=numel(x);
if n==0, q=NaN; return; end
r=1+(n-1)*p/100; lo=floor(r); hi=ceil(r);
if lo==hi, q=x(lo); else, q=x(lo)+(r-lo)*(x(hi)-x(lo)); end
end

function printMetrics(name,M)
fprintf('%s: Mean %.2f | Median %.2f | RMSE %.2f | P68 %.2f | P95 %.2f | Max %.2f | N %d\n', ...
    name,M.MeanPE_m,M.MedianPE_m,M.RMSE_m,M.P68_m,M.P95_m,M.MaxPE_m,M.N);
end
