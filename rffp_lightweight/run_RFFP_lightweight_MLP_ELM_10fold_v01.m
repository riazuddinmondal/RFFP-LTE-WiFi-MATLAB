function report = run_RFFP_lightweight_MLP_ELM_10fold_v01(dataFile,foldFile,outputDir)
% Ten-fold held-out-log compact MLP + ELM experiment, MATLAB R2024a.
% Dependencies on path: prepareFoldRepresentation, materializeAlignedRFFP,
% makeRFFPInnerValidation; makeRouteGroupedFolds recommended.
% Usage: report=run_RFFP_lightweight_MLP_ELM_10fold_v01;
% To resume: run the same command with the same output directory.
if nargin<1 || isempty(dataFile),dataFile='RFFP_corrected_representations.mat';end
if nargin<2 || isempty(foldFile),foldFile='corrected_route10fold_checkpoint.mat';end
if nargin<3 || isempty(outputDir),outputDir='Lightweight_10Fold_v01';end
assert(isfile(dataFile),'Missing dataset: %s',dataFile);
assert(isfile(foldFile),'Missing fold checkpoint: %s',foldFile);
assert(exist('prepareFoldRepresentation','file')==2,'Missing prepareFoldRepresentation.m');
assert(exist('materializeAlignedRFFP','file')==2,'Missing materializeAlignedRFFP.m');
assert(exist('makeRFFPInnerValidation','file')==2,'Missing makeRFFPInnerValidation.m');
if ~exist(outputDir,'dir'),mkdir(outputDir);end
cfg=struct();cfg.seed=20261117;cfg.mlpWidths={[128 64],[256 128]};
cfg.mlpMaxEpochs=80;cfg.mlpBatchSize=512;cfg.mlpLearningRate=1e-3;
cfg.mlpL2=1e-4;cfg.elmWidths=[256 512 1024];
cfg.elmRidge=[1e-2 1e-1 1];cfg.elmBatchSize=2048;
cfg.predBatchSize=2048;cfg.innerSeedBase=1042;
base=struct('MinTrainingObservations',50,'MaxLTEFeatures',128, ...
    'MaxWLANFeatures',384,'LTEMissingFloor_dBm',-140, ...
    'WLANMissingFloor_dBm',-105);
S=load(dataFile,'reps');assert(isfield(S,'reps'),'Expected reps struct');
reps=S.reps;clear S;
Y=double(reps.targetXY_m);N=size(Y,1);
assert(size(Y,2)==2 && all(isfinite(Y(:))),'Invalid coordinate targets');
assert(height(reps.metadata)==N && ismember('LogID',reps.metadata.Properties.VariableNames));
logID=reps.metadata.LogID;
F=load(foldFile);fold=extractFoldVector(F,N);clear F;
assert(all(ismember(unique(fold),1:10)) && numel(unique(fold))==10, ...
    'Expected 10 original outer folds numbered 1 through 10');
if exist('makeRouteGroupedFolds','file')==2
    expected=makeRouteGroupedFolds(reps.metadata,10,42);
    assert(isequal(double(expected(:)),fold(:)), ...
        'Fold checkpoint differs from original DNN seed-42 route partition');
end
% Refuse to silently reuse results produced under a different protocol.
manifest=fullfile(outputDir,'experiment_manifest.mat');
if isfile(manifest)
    M=load(manifest,'savedCfg','savedFold','savedN','savedDataFile','savedFoldFile');
    assert(isequaln(M.savedCfg,cfg) && isequal(M.savedFold,fold) && M.savedN==N, ...
        'Resume manifest differs from this run. Use a new output directory.');
    assert(strcmp(M.savedDataFile,whichOrPath(dataFile)) && ...
        strcmp(M.savedFoldFile,whichOrPath(foldFile)), ...
        'Dataset/checkpoint paths changed; use a new output directory.');
else
    savedCfg=cfg;savedFold=fold;savedN=N;
    savedDataFile=whichOrPath(dataFile);savedFoldFile=whichOrPath(foldFile);
    save(manifest,'savedCfg','savedFold','savedN','savedDataFile','savedFoldFile','-v7.3');
end
fprintf('RFFP compact MLP + ELM | 10 outer folds | N=%d\n',N);
for k=1:10
    resultPath=fullfile(outputDir,sprintf('fold_%02d_result.mat',k));
    if isfile(resultPath)
        R=load(resultPath,'foldResult');
        assert(isfield(R,'foldResult') && R.foldResult.Fold==k && ...
            isequal(R.foldResult.TestIndices(:),find(fold==k)), ...
            'Saved fold %d does not match current checkpoint',k);
        fprintf('Fold %d/10 already saved; skipping.\n',k);continue;
    end
    rng(cfg.seed,'twister');
    outerTest=fold==k;outerDev=~outerTest;
    assert(~isempty(find(outerTest,1)) && ~isempty(find(outerDev,1)));
    assert(isempty(intersect(unique(logID(outerTest)),unique(logID(outerDev)))), ...
        'LogID leakage in outer fold %d',k);
    [innerFit,innerVal,innerRoutes]=makeRFFPInnerValidation( ...
        reps.metadata,outerDev,5,1,42+1000+k);
    assert(all((innerFit|innerVal)==outerDev) && ~any(innerFit&innerVal), ...
        'Invalid inner split on fold %d',k);
    assert(isempty(intersect(unique(logID(innerFit)),unique(logID(innerVal)))), ...
        'LogID leakage in inner fold %d',k);
    fprintf('\n=== Fold %d/10 | dev %d | inner fit %d | val %d | test %d ===\n', ...
        k,nnz(outerDev),nnz(innerFit),nnz(innerVal),nnz(outerTest));
    tFold=tic;
    [XinFit,XinVal,innerFeatureInfo]=prepareFoldRepresentation( ...
        reps,"C",innerFit,innerVal,base);
    [coordInner,ZinFit]=fitCoordinates(Y(innerFit,:));
    ZinVal=applyCoordinates(Y(innerVal,:),coordInner);
    mlpCandidates=struct('widths',{},'valMean_m',{},'valP95_m',{}, ...
        'trainSeconds',{},'selectedEpoch',{});
    for c=1:numel(cfg.mlpWidths)
        widths=cfg.mlpWidths{c};rng(cfg.seed+100+c,'twister');
        t=tic;[net,info]=trainCompactMLP(XinFit,ZinFit,XinVal,ZinVal,widths,cfg);
        trainSeconds=toc(t);
        pred=undoCoordinates(predict(net,XinVal,'MiniBatchSize',cfg.predBatchSize),coordInner);
        met=metrics(Y(innerVal,:),pred);
        mlpCandidates(c)=struct('widths',widths,'valMean_m',met.Mean_m, ...
            'valP95_m',met.P95_m,'trainSeconds',trainSeconds, ...
            'selectedEpoch',bestEpoch(info));
        fprintf('Fold %d MLP [%s]: validation mean %.3f m, P95 %.3f m, %.1f s\n', ...
            k,num2str(widths),met.Mean_m,met.P95_m,trainSeconds);
        clear net info pred
    end
    [~,bestMLP]=min([mlpCandidates.valMean_m]);chosenMLP=mlpCandidates(bestMLP);
    elmCandidates=struct('hidden',{},'ridge',{},'valMean_m',{}, ...
        'valP95_m',{},'fitSeconds',{});c=0;
    for h=cfg.elmWidths
        rng(cfg.seed+200+h,'twister');d=size(XinFit,2);
        W=single(randn(d,h)/sqrt(double(d)));b=single(0.1*randn(1,h));
        t=tic;[G,C]=elmMoments(XinFit,ZinFit,W,b,cfg.elmBatchSize);
        for lambda=cfg.elmRidge
            c=c+1;B=elmSolve(G,C,lambda);
            elm=struct('W',W,'b',b,'B',B,'lambda',lambda,'hidden',h);
            pred=undoCoordinates(elmPredict(elm,XinVal,cfg.predBatchSize),coordInner);
            met=metrics(Y(innerVal,:),pred);
            elmCandidates(c)=struct('hidden',h,'ridge',lambda, ...
                'valMean_m',met.Mean_m,'valP95_m',met.P95_m,'fitSeconds',toc(t));
            fprintf('Fold %d ELM h=%d ridge=%g: validation mean %.3f m\n', ...
                k,h,lambda,met.Mean_m);
        end
        clear W b G C B elm pred
    end
    [~,bestELM]=min([elmCandidates.valMean_m]);chosenELM=elmCandidates(bestELM);
    clear XinFit XinVal ZinFit ZinVal
    [Xdev,Xtest,featureInfo]=prepareFoldRepresentation( ...
        reps,"C",outerDev,outerTest,base);
    [coordFinal,Zdev]=fitCoordinates(Y(outerDev,:));Ytest=Y(outerTest,:);
    finalEpoch=max(5,min(cfg.mlpMaxEpochs,chosenMLP.selectedEpoch));
    rng(cfg.seed+500+bestMLP,'twister');t=tic;
    mlpFinal=trainCompactMLPFinal(Xdev,Zdev,chosenMLP.widths,finalEpoch,cfg);
    mlpFitSeconds=toc(t);t=tic;
    mlpPred=undoCoordinates(predict(mlpFinal,Xtest, ...
        'MiniBatchSize',cfg.predBatchSize),coordFinal);
    mlpInferenceSeconds=toc(t);mlpMetrics=metrics(Ytest,mlpPred);
    rng(cfg.seed+200+chosenELM.hidden,'twister');
    d=size(Xdev,2);h=chosenELM.hidden;
    W=single(randn(d,h)/sqrt(double(d)));b=single(0.1*randn(1,h));
    t=tic;[G,C]=elmMoments(Xdev,Zdev,W,b,cfg.elmBatchSize);
    B=elmSolve(G,C,chosenELM.ridge);
    elmFinal=struct('W',W,'b',b,'B',B,'lambda',chosenELM.ridge,'hidden',h);
    elmFitSeconds=toc(t);t=tic;
    elmPred=undoCoordinates(elmPredict(elmFinal,Xtest,cfg.predBatchSize),coordFinal);
    elmInferenceSeconds=toc(t);elmMetrics=metrics(Ytest,elmPred);
    foldResult=struct('Fold',k,'TestIndices',find(outerTest), ...
        'TestLogID',logID(outerTest),'TestReference',Ytest, ...
        'MLPPredictions',mlpPred,'ELMPredictions',elmPred, ...
        'MLP',struct('Metrics',mlpMetrics,'FinalEpochs',finalEpoch, ...
            'FitSeconds',mlpFitSeconds,'InferenceSeconds',mlpInferenceSeconds), ...
        'ELM',struct('Metrics',elmMetrics,'FitSeconds',elmFitSeconds, ...
            'InferenceSeconds',elmInferenceSeconds), ...
        'MLPCandidates',mlpCandidates,'ELMCandidates',elmCandidates, ...
        'MLPSelected',chosenMLP,'ELMSelected',chosenELM, ...
        'InnerFitCount',nnz(innerFit),'InnerValCount',nnz(innerVal), ...
        'InnerValidationLogs',unique(logID(innerVal)), ...
        'InnerRouteTable',innerRoutes,'InnerFeatureInfo',innerFeatureInfo, ...
        'FeatureInfo',featureInfo,'ElapsedSeconds',toc(tFold));
    % Write only completed folds; no partial checkpoint is treated as complete.
    save(resultPath,'foldResult','-v7.3');
    fprintf('Fold %d COMPLETE: MLP mean %.3f P95 %.3f | ELM mean %.3f P95 %.3f | %.1f s\n', ...
        k,mlpMetrics.Mean_m,mlpMetrics.P95_m,elmMetrics.Mean_m,elmMetrics.P95_m,foldResult.ElapsedSeconds);
    clear mlpFinal elmFinal Xdev Xtest Zdev W b G C B
    % Refresh aggregate files after every fold; robust to interrupted runs.
    report=aggregateCompleted(outputDir,fold,Y,logID,cfg,dataFile,foldFile);
end
report=aggregateCompleted(outputDir,fold,Y,logID,cfg,dataFile,foldFile);
fprintf('\nCompleted %d of 10 folds. Output: %s\n',numel(report.CompletedFolds),outputDir);
if numel(report.CompletedFolds)==10
    disp(report.PooledSummary);
else
    fprintf('Pooled summary withheld until all 10 folds complete.\n');
end
end

function report=aggregateCompleted(outputDir,fold,Y,logID,cfg,dataFile,foldFile)
N=size(Y,1);mlpOOF=nan(N,2);elmOOF=nan(N,2);
rows=cell(0,1);completed=[];foldResults=cell(10,1);
for k=1:10
    f=fullfile(outputDir,sprintf('fold_%02d_result.mat',k));
    if ~isfile(f),continue;end
    S=load(f,'foldResult');r=S.foldResult;
    ix=find(fold==k);
    assert(r.Fold==k && isequal(r.TestIndices(:),ix), ...
        'Checkpoint mismatch in fold %d',k);
    assert(isequal(r.TestReference,Y(ix,:)),'Reference mismatch in fold %d',k);
    assert(isequal(r.TestLogID,logID(ix)),'LogID mismatch in fold %d',k);
    assert(size(r.MLPPredictions,1)==numel(ix) && ...
        size(r.ELMPredictions,1)==numel(ix),'Prediction count mismatch');
    mlpOOF(ix,:)=r.MLPPredictions;elmOOF(ix,:)=r.ELMPredictions;
    completed(end+1)=k; %#ok<AGROW>
    foldResults{k}=r;
    rows{end+1,1}=struct('Fold',k,'N',numel(ix), ...
        'MLPMean_m',r.MLP.Metrics.Mean_m,'MLPP68_m',r.MLP.Metrics.P68_m, ...
        'MLPP95_m',r.MLP.Metrics.P95_m,'MLPRMSE_m',r.MLP.Metrics.RMSE_m, ...
        'MLPFit_s',r.MLP.FitSeconds,'MLPInference_s',r.MLP.InferenceSeconds, ...
        'ELMMean_m',r.ELM.Metrics.Mean_m,'ELMP68_m',r.ELM.Metrics.P68_m, ...
        'ELMP95_m',r.ELM.Metrics.P95_m,'ELMRMSE_m',r.ELM.Metrics.RMSE_m, ...
        'ELMFit_s',r.ELM.FitSeconds,'ELMInference_s',r.ELM.InferenceSeconds); %#ok<AGROW>
end
report=struct('Config',cfg,'DataFile',dataFile,'FoldFile',foldFile, ...
    'CompletedFolds',completed,'FoldResults',{foldResults}, ...
    'MLPOOF',mlpOOF,'ELMOOF',elmOOF,'OuterFold',fold);
if ~isempty(rows)
    T=struct2table(vertcat(rows{:}));report.FoldSummary=T;
    writetable(T,fullfile(outputDir,'fold_summary.csv'));
else
    report.FoldSummary=table();
end
if numel(completed)==10
    assert(all(isfinite(mlpOOF(:))) && all(isfinite(elmOOF(:))), ...
        'Missing/nonfinite OOF predictions');
    m=metrics(Y,mlpOOF);e=metrics(Y,elmOOF);
    report.PooledMLP=m;report.PooledELM=e;
    report.PooledSummary=table({'CompactMLP';'ELM'}, ...
        [m.Mean_m;e.Mean_m],[m.Median_m;e.Median_m], ...
        [m.RMSE_m;e.RMSE_m],[m.P68_m;e.P68_m], ...
        [m.P95_m;e.P95_m],[m.Max_m;e.Max_m], ...
        [m.Coverage;e.Coverage], ...
        'VariableNames',{'Model','Mean_m','Median_m','RMSE_m', ...
        'P68_m','P95_m','Max_m','Coverage'});
    writetable(report.PooledSummary,fullfile(outputDir,'pooled_summary.csv'));
    % Save errors separately for CDF and paired bootstrap analyses.
    report.MLPErrors_m=sqrt(sum((Y-mlpOOF).^2,2));
    report.ELMErrors_m=sqrt(sum((Y-elmOOF).^2,2));
    report.LogID=logID;
end
save(fullfile(outputDir,'lightweight_10fold_combined.mat'),'report','-v7.3');
end

function p=whichOrPath(f)
q=which(f);
if isempty(q),p=char(string(f));else,p=q;end
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
