function report = run_RFFP_lightweight_MLP_ELM_fold1_v03(dataFile, foldFile, outputDir)
% RUN_RFFP_LIGHTWEIGHT_MLP_ELM_FOLD1_V01
% Exploratory, leakage-controlled Fold-1 compact MLP and ELM comparison.
%
% USAGE:
%   report = run_RFFP_lightweight_MLP_ELM_fold1_v01( ...
%       'RFFP_corrected_representations.mat', ...
%       'corrected_route10fold_checkpoint.mat', 'Lightweight_Fold1');
%
% REQUIRES ORIGINAL RFFP HELPERS ON MATLAB PATH:
%   prepareFoldRepresentation.m, makeRFFPInnerValidation.m,
%   makeRouteGroupedFolds.m and their dependencies.
% Loads reps.targetXY_m and reps.metadata from original MAT dataset.
% Uses original feature materialization and saved outer fold vector.
%
% FAIRNESS / INTERPRETATION:
%   - Outer Fold 1 is never used for feature selection, model selection or fitting.
%   - An inner LOG-GROUPED holdout is used for both MLP and ELM selection.
%   - Fixed candidate lists; no decisions based on Fold-1 test performance.
%   - Fold-1 results are exploratory, not a replacement for 10-fold evidence.
%   - ELM ridge uses hidden-feature moments; no N-by-N kernel matrix.
%   - MATLAB Deep Learning Toolbox required for compact MLP.
%
% Authoring: reproducible reference implementation; verify field mappings.

if nargin < 1 || isempty(dataFile), dataFile = 'RFFP_corrected_representations.mat'; end
if nargin < 2 || isempty(foldFile), foldFile = 'corrected_route10fold_checkpoint.mat'; end
if nargin < 3 || isempty(outputDir), outputDir = 'Lightweight_Fold1_v03'; end
assert(isfile(dataFile), 'Data MAT file not found: %s', dataFile);
assert(isfile(foldFile), 'Fold MAT file not found: %s', foldFile);
if ~exist(outputDir,'dir'), mkdir(outputDir); end

cfg.outerFold = 1;
cfg.seed = 20261117;
cfg.maxLTE = 128;
cfg.maxWiFi = 384;
cfg.minObservations = 50;
cfg.mlpWidths = {[128 64], [256 128]};
cfg.mlpMaxEpochs = 80;
cfg.mlpBatchSize = 512;
cfg.mlpLearningRate = 1e-3;
cfg.mlpL2 = 1e-4;
cfg.elmWidths = [256 512 1024];
cfg.elmRidge = [1e-2 1e-1 1];
cfg.elmBatchSize = 2048;
cfg.predBatchSize = 2048;
cfg.verbose = true;

rng(cfg.seed,'twister');
fprintf('Lightweight RFFP Fold %d | MLP and ELM | seed %d\n',cfg.outerFold,cfg.seed);
% Reuse the EXACT feature and fold helpers from the original tuned DNN.
assert(exist('prepareFoldRepresentation','file')==2, ...
    'Add the original rffp_10fold folder containing prepareFoldRepresentation.m to the MATLAB path.');
assert(exist('makeRFFPInnerValidation','file')==2, ...
    'Add the rffp_dnn folder containing makeRFFPInnerValidation.m to the MATLAB path.');
S=load(dataFile,'reps');
assert(isfield(S,'reps'),'Expected reps in the original RFFP representation file.');
reps=S.reps; clear S;
assert(isfield(reps,'targetXY_m') && isfield(reps,'metadata'), ...
    'Expected reps.targetXY_m and reps.metadata from original DNN.');
Y=double(reps.targetXY_m); N=size(Y,1);
F=load(foldFile);
fold=extractFoldVector(F,N); clear F;
if exist('makeRouteGroupedFolds','file')==2
    expected=makeRouteGroupedFolds(reps.metadata,10,42);
    assert(isequal(double(expected(:)),fold(:)), ...
        'Saved outer fold vector differs from original DNN seed-42 route partition.');
end
assert(isequal(size(Y),[N,2]) && all(isfinite(Y(:))));
assert(height(reps.metadata)==N && ismember('LogID',reps.metadata.Properties.VariableNames));
logID=reps.metadata.LogID;
outerTest=fold==cfg.outerFold; outerDev=~outerTest;
assert(nnz(outerTest)>0 && nnz(outerDev)>0);
assert(isempty(intersect(unique(logID(outerTest)),unique(logID(outerDev)))), ...
    'Outer fold leaks LogID between development and test.');
% Match the tuned DNN's five-way grouped inner split, seed 42+1000+fold.
[innerFit,innerVal,innerRoutes]=makeRFFPInnerValidation( ...
    reps.metadata,outerDev,5,1,42+1000+cfg.outerFold);
assert(~any(innerFit & innerVal) && all(innerFit|innerVal==outerDev));
base=struct('MinTrainingObservations',50,'MaxLTEFeatures',128, ...
    'MaxWLANFeatures',384,'LTEMissingFloor_dBm',-140, ...
    'WLANMissingFloor_dBm',-105);
[XinFit,XinVal,innerFeatureInfo]=prepareFoldRepresentation( ...
    reps,"C",innerFit,innerVal,base);
[coordInner,ZinFit]=fitCoordinates(Y(innerFit,:));
ZinVal=applyCoordinates(Y(innerVal,:),coordInner);
fprintf('N=%d | development=%d | fit=%d | validation=%d | test=%d\n', ...
    N,nnz(outerDev),nnz(innerFit),nnz(innerVal),nnz(outerTest));

%% COMPACT MLP: select by validation mean positioning error in metres.
mlpCandidates=struct('widths',{},'valMean_m',{},'valP95_m',{},'trainSeconds',{},'selectedEpoch',{});
mlpModels=cell(numel(cfg.mlpWidths),1);
for c=1:numel(cfg.mlpWidths)
    widths=cfg.mlpWidths{c};
    fprintf('MLP candidate %d/%d: [%s]\n',c,numel(cfg.mlpWidths),num2str(widths));
    rng(cfg.seed+100+c,'twister');
    t=tic;
    [net,info]=trainCompactMLP(XinFit,ZinFit,XinVal,ZinVal,widths,cfg);
    elapsed=toc(t);
    zhat=predict(net,XinVal,'MiniBatchSize',cfg.predBatchSize);
    yhat=undoCoordinates(zhat,coordInner);
    met=metrics(Y(innerVal,:),yhat);
    mlpModels{c}=net;
    mlpCandidates(c)=struct('widths',widths,'valMean_m',met.Mean_m, ...
        'valP95_m',met.P95_m,'trainSeconds',elapsed, ...
        'selectedEpoch',bestEpoch(info));
    fprintf('  val mean %.3f m | P95 %.3f m | fit %.1f s\n',met.Mean_m,met.P95_m,elapsed);
end
[~,bestMLP]=min([mlpCandidates.valMean_m]);
chosenMLP=mlpCandidates(bestMLP);

%% ELM: frozen random ReLU features, ridge fit, validation selection.
elmCandidates=struct('hidden',{},'ridge',{},'valMean_m',{},'valP95_m',{},'fitSeconds',{});
elmCandidateModels=cell(numel(cfg.elmWidths)*numel(cfg.elmRidge),1);
c=0;
for h=cfg.elmWidths
    rng(cfg.seed+200+h,'twister');
    inputD=size(XinFit,2);
    W=single(randn(inputD,h)/sqrt(double(inputD)));
    b=single(0.1*randn(1,h));
    t=tic;
    [G,C]=elmMoments(XinFit,ZinFit,W,b,cfg.elmBatchSize);
    for lambda=cfg.elmRidge
        c=c+1;
        B=elmSolve(G,C,lambda);
        elm=struct('W',W,'b',b,'B',B,'lambda',lambda,'hidden',h);
        zhat=elmPredict(elm,XinVal,cfg.predBatchSize);
        yhat=undoCoordinates(zhat,coordInner);
        met=metrics(Y(innerVal,:),yhat);
        elmCandidates(c)=struct('hidden',h,'ridge',lambda, ...
            'valMean_m',met.Mean_m,'valP95_m',met.P95_m,'fitSeconds',toc(t));
        elmCandidateModels{c}=elm;
        fprintf('ELM h=%d lambda=%g | val mean %.3f m | P95 %.3f m\n', ...
            h,lambda,met.Mean_m,met.P95_m);
    end
end
[~,bestELM]=min([elmCandidates.valMean_m]);
chosenELM=elmCandidates(bestELM);

%% FINAL REFIT: all outer development logs, test untouched until now.
% Refit feature selection and scalers using all outer development samples.
[Xdev,Xtest,featureInfo]=prepareFoldRepresentation( ...
    reps,"C",outerDev,outerTest,base);
[coordFinal,Zdev]=fitCoordinates(Y(outerDev,:));
Ytest=Y(outerTest,:);

% MLP final refit uses inner best-validation epoch, not test-based stopping.
finalEpoch=max(5,min(cfg.mlpMaxEpochs,chosenMLP.selectedEpoch));
rng(cfg.seed+500+bestMLP,'twister');
t=tic;
mlpFinal=trainCompactMLPFinal(Xdev,Zdev,chosenMLP.widths,finalEpoch,cfg);
mlpFitSeconds=toc(t);
t=tic;
mlpPred=undoCoordinates(predict(mlpFinal,Xtest,'MiniBatchSize',cfg.predBatchSize),coordFinal);
mlpInferenceSeconds=toc(t);
mlpMetrics=metrics(Ytest,mlpPred);

rng(cfg.seed+200+chosenELM.hidden,'twister');
D=size(Xdev,2); H=chosenELM.hidden;
W=single(randn(D,H)/sqrt(double(D))); b=single(0.1*randn(1,H));
t=tic;
[G,C]=elmMoments(Xdev,Zdev,W,b,cfg.elmBatchSize);
B=elmSolve(G,C,chosenELM.ridge);
elmFinal=struct('W',W,'b',b,'B',B,'lambda',chosenELM.ridge,'hidden',H);
elmFitSeconds=toc(t);
t=tic;
elmPred=undoCoordinates(elmPredict(elmFinal,Xtest,cfg.predBatchSize),coordFinal);
elmInferenceSeconds=toc(t);
elmMetrics=metrics(Ytest,elmPred);

% Strictly preserve prediction order of the original N-by-1 outer-test mask.
report=struct(); report.Config=cfg; report.DataFile=dataFile;
report.FoldFile=foldFile; report.OuterFold=cfg.outerFold;
report.TestIndices=find(outerTest); report.DevCount=nnz(outerDev);
report.InnerFitCount=nnz(innerFit); report.InnerValCount=nnz(innerVal);
report.InnerValidationLogs=unique(logID(innerVal));
report.InnerRouteTable=innerRoutes; report.InnerFeatureInfo=innerFeatureInfo;
report.FeatureInfo=featureInfo;
report.MLPCandidates=mlpCandidates; report.ELMCandidates=elmCandidates;
report.MLPSelected=chosenMLP; report.ELMSelected=chosenELM;
report.MLP=struct('Metrics',mlpMetrics,'FinalEpochs',finalEpoch, ...
    'FitSeconds',mlpFitSeconds,'InferenceSeconds',mlpInferenceSeconds);
report.ELM=struct('Metrics',elmMetrics,'FitSeconds',elmFitSeconds, ...
    'InferenceSeconds',elmInferenceSeconds);
report.TestReference=Ytest; report.MLPPredictions=mlpPred; report.ELMPredictions=elmPred;
report.TestLogID=logID(outerTest);
report.Note=['Exploratory Fold-1 results; do not select hyperparameters using test results. ', ...
    'Feature selection and validation splits are development-only.'];

save(fullfile(outputDir,'lightweight_fold1_report_v03.mat'),'report','-v7.3');
save(fullfile(outputDir,'lightweight_fold1_models_v03.mat'), ...
    'mlpFinal','elmFinal','featureInfo','coordFinal','-v7.3');
T=table({'CompactMLP';'ELM'},[mlpMetrics.Mean_m;elmMetrics.Mean_m], ...
    [mlpMetrics.Median_m;elmMetrics.Median_m],[mlpMetrics.RMSE_m;elmMetrics.RMSE_m], ...
    [mlpMetrics.P68_m;elmMetrics.P68_m],[mlpMetrics.P95_m;elmMetrics.P95_m], ...
    [mlpMetrics.Coverage;elmMetrics.Coverage],[mlpFitSeconds;elmFitSeconds], ...
    [mlpInferenceSeconds;elmInferenceSeconds], ...
    'VariableNames',{'Model','Mean_m','Median_m','RMSE_m','P68_m','P95_m', ...
    'Coverage','FinalFit_s','TestInference_s'});
writetable(T,fullfile(outputDir,'lightweight_fold1_summary_v03.csv'));
disp(T);
fprintf('Saved results to: %s\n',outputDir);
end

function fold=extractFoldVector(F,N)
if isfield(F,'savedFoldIndex'), fold=F.savedFoldIndex;
elseif isfield(F,'foldIndex'), fold=F.foldIndex;
elseif isfield(F,'fold'), fold=F.fold;
else
    fprintf('Fold MAT variables: %s\n',strjoin(fieldnames(F)',', '));
    error('Could not find savedFoldIndex, foldIndex, or fold.');
end
fold=double(fold(:)); assert(numel(fold)==N,'Fold vector length does not match X.');
end

function [s,Z]=fitCoordinates(Y)
s=struct('mu',mean(Y,1),'sigma',std(Y,0,1));
s.sigma(~isfinite(s.sigma)|s.sigma==0)=1;
Z=single((Y-s.mu)./s.sigma);
end
function Z=applyCoordinates(Y,s), Z=single((Y-s.mu)./s.sigma); end
function Y=undoCoordinates(Z,s), Y=double(Z).*s.sigma+s.mu; end

function layers=makeMLPLayers(d,widths)
layers=[featureInputLayer(d,'Normalization','none','Name','features')];
for k=1:numel(widths)
    layers=[layers;fullyConnectedLayer(widths(k),'Name',sprintf('fc%d',k)); ...
        reluLayer('Name',sprintf('relu%d',k))]; %#ok<AGROW>
end
layers=[layers;fullyConnectedLayer(2,'Name','coordinate_output'); ...
    regressionLayer('Name','regression')];
end

function [net,info]=trainCompactMLP(X,Z,Xv,Zv,widths,cfg)
opts=trainingOptions('adam','InitialLearnRate',cfg.mlpLearningRate, ...
    'MaxEpochs',cfg.mlpMaxEpochs,'MiniBatchSize',cfg.mlpBatchSize, ...
    'L2Regularization',cfg.mlpL2,'Shuffle','every-epoch', ...
    'ValidationData',{Xv,Zv},'ValidationFrequency',max(1,ceil(size(X,1)/cfg.mlpBatchSize)), ...
    'ValidationPatience',10,'OutputNetwork','best-validation-loss', ...
    'Verbose',false,'Plots','none','ExecutionEnvironment','auto');
[net,info]=trainNetwork(X,Z,makeMLPLayers(size(X,2),widths),opts);
end

function net=trainCompactMLPFinal(X,Z,widths,epochs,cfg)
opts=trainingOptions('adam','InitialLearnRate',cfg.mlpLearningRate, ...
    'MaxEpochs',epochs,'MiniBatchSize',cfg.mlpBatchSize, ...
    'L2Regularization',cfg.mlpL2,'Shuffle','every-epoch', ...
    'Verbose',false,'Plots','none','ExecutionEnvironment','auto');
net=trainNetwork(X,Z,makeMLPLayers(size(X,2),widths),opts);
end

function e=bestEpoch(info)
% Epoch associated with minimum finite validation loss.
e=20;
if isfield(info,'ValidationLoss')
    v=info.ValidationLoss(:); valid=find(isfinite(v));
    if ~isempty(valid)
        [~,j]=min(v(valid)); idx=valid(j);
        if isfield(info,'Epoch') && numel(info.Epoch)>=idx
            e=max(1,round(info.Epoch(idx)));
        else
            e=max(1,round(idx));
        end
    end
end
end

function [G,C]=elmMoments(X,Z,W,b,batchSize)
H=size(W,2); G=zeros(H+1,H+1); C=zeros(H+1,2);
for i=1:batchSize:size(X,1)
    ii=i:min(i+batchSize-1,size(X,1));
    A=double(max(0,single(X(ii,:))*W+b));
    A=[A,ones(numel(ii),1)];
    G=G+A'*A; C=C+A'*double(Z(ii,:));
end
end
function B=elmSolve(G,C,lambda)
R=eye(size(G)); R(end,end)=0; % do not penalize output intercept
B=(G+lambda*R)\C;
end
function Z=elmPredict(model,X,batchSize)
Z=zeros(size(X,1),2);
for i=1:batchSize:size(X,1)
    ii=i:min(i+batchSize-1,size(X,1));
    A=double(max(0,single(X(ii,:))*model.W+model.b));
    Z(ii,:)=[A,ones(numel(ii),1)]*model.B;
end
end
function m=metrics(Y,Yhat)
valid=all(isfinite(Yhat),2);
e=sqrt(sum((Y(valid,:)-Yhat(valid,:)).^2,2));
assert(~isempty(e),'No finite predictions.');
m=struct('Mean_m',mean(e),'Median_m',median(e), ...
    'RMSE_m',sqrt(mean(e.^2)),'P68_m',prctile(e,68), ...
    'P95_m',prctile(e,95),'Max_m',max(e), ...
    'Coverage',nnz(valid)/size(Y,1),'Errors_m',e);
end
