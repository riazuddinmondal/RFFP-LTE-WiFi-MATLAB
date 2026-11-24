function results = run_RFFP_tuned_DNN_review_validation(dataFile,foldFile,mode,foldsToRun,outputDir)
% Tuned DNN: original saved 10-fold evaluation OR directional cross-campaign transfer.
% MATLAB R2024a. Run one fold first. No held-out labels used for tuning.
% Requires uploaded buildRFFPMLP.m, makeRFFPDNNCandidates.m,
% evaluateRFFPDNNCandidate.m, trainFinalRFFPDNN.m on MATLAB path.
% Crosscampaign fold 1: train 2014/test 2015; fold 2: reverse.
% Original10 mode: uses savedFoldIndex, NOT newly generated folds.
if nargin<1 || isempty(dataFile), dataFile='RFFP_corrected_representations.mat';end
if nargin<2 || isempty(foldFile), foldFile='corrected_route10fold_checkpoint.mat';end
if nargin<3 || isempty(mode), mode='crosscampaign';end
if nargin<4 || isempty(foldsToRun), foldsToRun=1;end
if nargin<5 || isempty(outputDir), outputDir=fullfile(pwd,'DNN_review_validation');end
if ~isfolder(outputDir),mkdir(outputDir);end
assert(ismember(lower(string(mode)),["crosscampaign","original10"]),'Unknown mode.');
S=load(dataFile,'reps','parsed'); reps=S.reps;
Y=double(reps.targetXY_m); N=size(Y,1);
assert(size(Y,2)==2 && all(isfinite(Y),'all'),'Invalid target coordinates');
logID=double(reps.metadata.LogID(:)); assert(numel(logID)==N);
wlanStart=double(S.parsed.WLANIDStart);
year=2015*ones(N,1); year(logID>=22 & logID<=37)=2014;
assert(all(ismember(logID,0:37)),'Unknown LogIDs: verify campaign mapping.');
assert(nnz(year==2014)==32798 && nnz(year==2015)==86132, ...
 'Campaign row counts differ from verified mapping. Stop and investigate.');
campaignRows2014=find(year==2014);campaignRows2015=find(year==2015);
save(fullfile(outputDir,'campaign_partition_indices.mat'), ...
 'campaignRows2014','campaignRows2015','logID','year','-v7.3');
if strcmpi(mode,'original10')
 F=load(foldFile,'savedFoldIndex'); assert(isfield(F,'savedFoldIndex'));
 foldIndex=double(F.savedFoldIndex(:));
 assert(numel(foldIndex)==N && isequal(unique(foldIndex),(1:10)'));
 for id=unique(logID)'
  assert(numel(unique(foldIndex(logID==id)))==1,'LogID split across folds');
 end
 assert(all(ismember(foldsToRun,1:10)));
else
 foldIndex=[];assert(all(ismember(foldsToRun,1:2)));
end
base=struct();
base.MinTrainingObservations=50; base.MaxLTEFeatures=128;
base.MaxWLANFeatures=384; base.LTEMissingFloor_dBm=-140;
base.WLANMissingFloor_dBm=-105;
base.HiddenUnits=[512 256 128 64]; base.Dropout=[.20 .15 .10];
base.InitialLearnRate=1e-3; base.LearnRateDropFactor=.5;
base.LearnRateDropPeriod=15; base.L2Regularization=1e-4;
base.MiniBatchSize=512; base.PredictionMiniBatchSize=2048;
base.MaxValidationEpochs=100; base.ValidationPatience=12;
base.MinimumFinalEpochs=5; base.ExecutionEnvironment='auto';
base.TailWeight=.10; base.RandomSeed=42; base.Verbose=false;
candidates=makeRFFPDNNCandidates(base);
assert(numel(candidates)==10);
results=struct([]);
for fold=foldsToRun(:)'
 if strcmpi(mode,'crosscampaign')
  if fold==1, outerTrain=year==2014;outerTest=year==2015;
  else,outerTrain=year==2015;outerTest=year==2014;end
 else
  outerTrain=foldIndex~=fold;outerTest=foldIndex==fold;
 end
 key=sprintf('%s_F%02d',lower(mode),fold);
 folder=fullfile(outputDir,key);if ~isfolder(folder),mkdir(folder);end
 doneFile=fullfile(folder,'completed_result.mat');
 if isfile(doneFile)
  D=load(doneFile,'row');fprintf('Already complete: %s\n',key);
  if isempty(results),results=D.row;else,results(end+1)=D.row;end
  continue;
 end
 assert(~any(outerTrain & outerTest));
 assert(isempty(intersect(unique(logID(outerTrain)),unique(logID(outerTest)))));
 % Deterministic inner validation by complete LogID groups, entirely within outer train.
 [fitMask,valMask]=splitLogs(logID,outerTrain,5,42+1000+fold);
 fprintf('\n%s | outer train %d (%d logs) | test %d (%d logs) | inner fit %d | val %d\n', ...
  key,nnz(outerTrain),numel(unique(logID(outerTrain))),nnz(outerTest), ...
  numel(unique(logID(outerTest))),nnz(fitMask),nnz(valMask));
 [XFit,XVal,innerFeatureInfo]=prepareAlignedC(reps,fitMask,valMask,base,wlanStart);
 trials=struct([]);
 trialFile=fullfile(folder,'tuning_checkpoint.mat');
 if isfile(trialFile)
  C=load(trialFile,'trials');trials=C.trials;
 end
 for c=1:numel(candidates)
  if numel(trials)>=c && isfield(trials(c),'Complete') && trials(c).Complete
   fprintf('  Candidate %d/10 resumed\n',c);continue;
  end
  rng(42+100000*fold+c,'twister');t=tic;
  [score,M,epoch,info]=evaluateRFFPDNNCandidate(XFit,Y(fitMask,:),XVal,Y(valMask,:),candidates(c));
  r=struct('Complete',true,'Index',c,'Name',char(candidates(c).CandidateName), ...
   'Score',score,'Mean_m',M.Mean_m,'P95_m',M.P95_m,'RMSE_m',M.RMSE_m, ...
   'BestEpoch',epoch,'Runtime_s',toc(t),'Config',candidates(c),'TrainingInfo',info);
  if isempty(trials),trials=r;else,trials(c)=orderfields(r,trials);end
  save(trialFile,'trials','innerFeatureInfo','-v7.3');
  fprintf('  %2d/10 %-14s | score %.3f | mean %.3f | P95 %.3f | epoch %d | %.1fs\n', ...
   c,r.Name,score,r.Mean_m,r.P95_m,epoch,r.Runtime_s);
 end
 [~,best]=min([trials.Score]); selected=trials(best).Config;
 bestEpoch=trials(best).BestEpoch;
 clear XFit XVal
 [XTrain,XTest,featureInfo]=prepareAlignedC(reps,outerTrain,outerTest,base,wlanStart);
 rng(42+200000+fold,'twister');t=tic;
 [net,targetCenter,targetScale,trainingInfo]=trainFinalRFFPDNN( ...
  XTrain,Y(outerTrain,:),bestEpoch,selected);
 pn=predict(net,XTest,'MiniBatchSize',selected.PredictionMiniBatchSize);
 pred=double(pn).*targetScale+targetCenter;
 runtime=toc(t); YTest=Y(outerTest,:);
 errors=sqrt(sum((pred-YTest).^2,2));
 metrics=localMetrics(errors);
 testRows=find(outerTest);testLogID=logID(outerTest);testYear=year(outerTest);
 save(fullfile(folder,'predictions.mat'),'pred','YTest','errors','testRows','testLogID','testYear','-v7.3');
 save(fullfile(folder,'trained_model.mat'),'net','targetCenter','targetScale', ...
  'selected','bestEpoch','featureInfo','trainingInfo','-v7.3');
 row=struct('Mode',char(mode),'Fold',fold,'NumTrain',nnz(outerTrain), ...
  'NumTest',nnz(outerTest),'Candidate',char(selected.CandidateName), ...
  'BestEpoch',bestEpoch,'TuningRuntime_s',sum([trials.Runtime_s]), ...
  'FinalRuntime_s',runtime,'Metrics',metrics,'ResultFolder',folder);
 save(doneFile,'row','-v7.3');
 fprintf('\nFINAL %s | %s | mean %.3f | median %.3f | RMSE %.3f | P68 %.3f | P95 %.3f m\n', ...
  key,row.Candidate,metrics.Mean_m,metrics.Median_m,metrics.RMSE_m,metrics.P68_m,metrics.P95_m);
 if isempty(results),results=row;else,results(end+1)=row;end
 clear net XTrain XTest pred pn
end
save(fullfile(outputDir,sprintf('%s_summary.mat',lower(mode))),'results','-v7.3');
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

function [A,B,info]=prepareAlignedC(reps,fitMask,otherMask,cfg,wlanStart)
 X=reps.C_alignedSparse.PowerObserved_dBm;
 ids=double(reps.C_alignedSparse.TransmitterID(:));
 assert(size(X,1)==numel(fitMask),'Sparse matrix row count mismatch');
 % Parser-defined boundary is the authoritative LTE/WLAN identifier split.
 isLTE=ids<wlanStart;
 assert(any(isLTE)&&any(~isLTE),'Invalid LTE/WLAN identifier boundary');
 fitRows=find(fitMask);otherRows=find(otherMask);
 counts=full(sum(X(fitRows,:)~=0,1))';
 eligible=counts>=cfg.MinTrainingObservations;
 lte=find(isLTE & eligible); wifi=find(~isLTE & eligible);
 [~,p]=sortrows([-counts(lte),double(lte)]);lte=lte(p(1:min(numel(p),cfg.MaxLTEFeatures)));
 [~,p]=sortrows([-counts(wifi),double(wifi)]);wifi=wifi(p(1:min(numel(p),cfg.MaxWLANFeatures)));
 selected=[lte(:);wifi(:)];assert(~isempty(selected));
 % Two channels per radio: normalized RSS and binary detection indicator.
 % Missing values are replaced by the technology-specific fixed floor.
 floorDB=[repmat(cfg.LTEMissingFloor_dBm,numel(lte),1); ...
          repmat(cfg.WLANMissingFloor_dBm,numel(wifi),1)]';
 A=transformRows(X,fitRows,selected,floorDB);
 B=transformRows(X,otherRows,selected,floorDB);
 info=struct('SelectedColumns',selected,'SelectedTransmitterIDs',ids(selected), ...
  'NumLTE',numel(lte),'NumWLAN',numel(wifi),'NumFeatures',size(A,2), ...
  'NumFit',size(A,1),'NumOther',size(B,1));
 fprintf('  Selected %d LTE + %d WLAN radios, %d input features\n', ...
  numel(lte),numel(wifi),size(A,2));
end

function Z=transformRows(X,rows,cols,floorDB)
 raw=single(full(X(rows,cols)));detected=raw~=0;
 floorMat=repmat(single(floorDB),size(raw,1),1);
 raw(~detected)=floorMat(~detected);
 % Stable fixed RSS scaling; never estimate transforms using test rows.
 raw=max(-140,min(-35,raw));
 Z=[(raw+140)/105,single(detected)];
 assert(all(isfinite(Z),'all'));
end

function M=localMetrics(e)
 e=double(e(:));M=struct('N',numel(e),'Mean_m',mean(e), ...
 'Median_m',median(e),'RMSE_m',sqrt(mean(e.^2)), ...
 'P68_m',prctile(e,68),'P95_m',prctile(e,95),'Maximum_m',max(e));
end
