function reports=run_RFFP_compact_MLP_crosscampaign(dataFile,outputDir,directions)
% Same compact MLP candidate architectures and train-only feature preprocessing
% as run_RFFP_lightweight_MLP_ELM_fold1_v03. ELM intentionally omitted.
if nargin<1||isempty(dataFile),dataFile='RFFP_corrected_representations.mat';end
if nargin<2||isempty(outputDir),outputDir='CompactMLP_crosscampaign_results';end
if nargin<3||isempty(directions),directions=[1 2];end
assert(all(ismember(directions,[1 2])));
if ~isfolder(outputDir),mkdir(outputDir);end
assert(exist('prepareFoldRepresentation','file')==2, ...
 'Original prepareFoldRepresentation.m is required on MATLAB path');
S=load(dataFile,'reps');reps=S.reps;
Y=double(reps.targetXY_m);N=size(Y,1);
logID=double(reps.metadata.LogID(:));assert(numel(logID)==N);
year=2015*ones(N,1);year(logID>=22)=2014;
assert(all(ismember(logID,0:37))&&nnz(year==2014)==32798 ...
 &&nnz(year==2015)==86132,'Campaign mapping mismatch');
base=struct('MinTrainingObservations',50,'MaxLTEFeatures',128, ...
 'MaxWLANFeatures',384,'LTEMissingFloor_dBm',-140, ...
 'WLANMissingFloor_dBm',-105);
cfg=struct('seed',20261117,'mlpWidths',{{[128 64],[256 128]}}, ...
 'mlpMaxEpochs',80,'mlpBatchSize',512,'mlpLearningRate',1e-3, ...
 'mlpL2',1e-4,'predBatchSize',2048);
reports=struct([]);
for direction=directions(:)'
 if direction==1,trainYear=2014;testYear=2015;
 else,trainYear=2015;testYear=2014;end
 folder=fullfile(outputDir,sprintf('F%02d',direction));
 if ~isfolder(folder),mkdir(folder);end
 completed=fullfile(folder,'completed_result.mat');
 if isfile(completed)
  C=load(completed,'report');r=C.report;
  fprintf('Already completed direction %d\n',direction);
  if isempty(reports),reports=r;else,reports(end+1)=r;end %#ok<AGROW>
  continue;
 end
 trainMask=year==trainYear;testMask=year==testYear;
 assert(isempty(intersect(unique(logID(trainMask)),unique(logID(testMask)))));
 [fitMask,valMask]=splitLogs(logID,trainMask,5,42+1000+direction);
 [XinFit,XinVal,innerFeatureInfo]=prepareFoldRepresentation( ...
  reps,"C",fitMask,valMask,base);
 [coordInner,ZinFit]=fitCoordinates(Y(fitMask,:));
 ZinVal=applyCoordinates(Y(valMask,:),coordInner);
 candidates=struct([]);checkpoint=fullfile(folder,'tuning_checkpoint.mat');
 if isfile(checkpoint),C=load(checkpoint,'candidates');candidates=C.candidates;end
 for c=1:numel(cfg.mlpWidths)
  if numel(candidates)>=c && candidates(c).Complete,continue;end
  widths=cfg.mlpWidths{c};rng(cfg.seed+100+c,'twister');t=tic;
  [net,info]=trainCompactMLP(XinFit,ZinFit,XinVal,ZinVal,widths,cfg);
  pred=undoCoordinates(predict(net,XinVal,'MiniBatchSize',cfg.predBatchSize),coordInner);
  met=metrics(Y(valMask,:),pred);
  trial=struct('Complete',true,'Widths',widths,'ValMean_m',met.Mean_m, ...
   'ValP95_m',met.P95_m,'BestEpoch',bestEpoch(info),'Train_s',toc(t));
  if isempty(candidates),candidates=trial;else,candidates(c)=trial;end
  save(checkpoint,'candidates','innerFeatureInfo','-v7.3');
  fprintf('Direction %d candidate %d | validation mean %.3f P95 %.3f\n', ...
   direction,c,met.Mean_m,met.P95_m);
 end
 [~,best]=min([candidates.ValMean_m]);chosen=candidates(best);
 clear XinFit XinVal;
 [Xtrain,Xtest,featureInfo]=prepareFoldRepresentation( ...
  reps,"C",trainMask,testMask,base);
 [coordFinal,Ztrain]=fitCoordinates(Y(trainMask,:));
 finalEpoch=max(5,min(cfg.mlpMaxEpochs,chosen.BestEpoch));
 rng(cfg.seed+500+best,'twister');t=tic;
 mlpFinal=trainCompactMLPFinal(Xtrain,Ztrain,chosen.Widths,finalEpoch,cfg);
 fitSeconds=toc(t);t=tic;
 pred=undoCoordinates(predict(mlpFinal,Xtest, ...
  'MiniBatchSize',cfg.predBatchSize),coordFinal);
 inferenceSeconds=toc(t);
 met=metrics(Y(testMask,:),pred);
 report=struct('Direction',direction,'TrainingYear',trainYear, ...
  'TestYear',testYear,'NumTrain',nnz(trainMask),'NumTest',nnz(testMask), ...
  'NumInnerFit',nnz(fitMask),'NumInnerVal',nnz(valMask), ...
  'InnerValidationLogID',unique(logID(valMask)), ...
  'TestIndices',find(testMask),'TestLogID',logID(testMask), ...
  'InnerFeatureInfo',innerFeatureInfo,'FeatureInfo',featureInfo, ...
  'Candidates',candidates,'ChosenCandidate',best, ...
  'FinalEpochs',finalEpoch,'Metrics',met, ...
  'FitSeconds',fitSeconds,'InferenceSeconds',inferenceSeconds);
 save(fullfile(folder,'trained_model.mat'),'mlpFinal','featureInfo','coordFinal','-v7.3');
 errors=met.Errors_m;testRows=find(testMask);testLogID=logID(testMask);
 save(fullfile(folder,'predictions.mat'),'pred','errors','testRows','testLogID','-v7.3');
 save(completed,'report','-v7.3');
 fprintf('Compact MLP %d -> %d | mean %.3f P68 %.3f P95 %.3f | fit %.1fs infer %.1fs\n', ...
 trainYear,testYear,met.Mean_m,met.P68_m,met.P95_m,fitSeconds,inferenceSeconds);
 if isempty(reports),reports=report;else,reports(end+1)=report;end %#ok<AGROW>
end
end

function [fitMask,valMask]=splitLogs(logID,trainMask,k,seed)
 u=unique(logID(trainMask)); assert(numel(u)>=k,'Too few training logs');
 rng(seed,'twister');u=u(randperm(numel(u)));
 % Group-level deterministic partition; balance approximate observation counts.
 counts=arrayfun(@(id)nnz(trainMask & logID==id),u);
 [~,ord]=sort(counts,'descend');u=u(ord);counts=counts(ord);
 groupCount=zeros(k,1);assignment=zeros(numel(u),1);
 for j=1:numel(u)
  [~,g]=min(groupCount);assignment(j)=g;groupCount(g)=groupCount(g)+counts(j);
 end
 valLogs=u(assignment==1);
 valMask=trainMask & ismember(logID,valLogs);fitMask=trainMask & ~valMask;
 assert(any(fitMask)&&any(valMask));
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


function m=metrics(Y,Yhat)
valid=all(isfinite(Yhat),2);
e=sqrt(sum((Y(valid,:)-Yhat(valid,:)).^2,2));
assert(~isempty(e),'No finite predictions.');
m=struct('Mean_m',mean(e),'Median_m',median(e), ...
    'RMSE_m',sqrt(mean(e.^2)),'P68_m',prctile(e,68), ...
    'P95_m',prctile(e,95),'Max_m',max(e), ...
    'Coverage',nnz(valid)/size(Y,1),'Errors_m',e);
end
