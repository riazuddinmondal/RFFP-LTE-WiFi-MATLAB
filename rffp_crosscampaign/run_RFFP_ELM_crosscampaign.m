function reports = run_RFFP_ELM_crosscampaign(dataFile,outputDir,directions)
% Bidirectional cross-campaign ELM, adapted from
% run_RFFP_lightweight_MLP_ELM_fold1_v03.m (ELM section only).
% Requires prepareFoldRepresentation.m and materializeAlignedRFFP.m.
% Example: run_RFFP_ELM_crosscampaign('RFFP_corrected_representations.mat', ...
%                                    'ELM_crosscampaign_results',1:2)
if nargin<1 || isempty(dataFile),dataFile='RFFP_corrected_representations.mat';end
if nargin<2 || isempty(outputDir),outputDir='ELM_crosscampaign_results';end
if nargin<3 || isempty(directions),directions=1:2;end
assert(all(ismember(directions,[1 2])),'Directions must be 1 and/or 2.');
assert(isfile(dataFile),'Missing data file: %s',dataFile);
assert(exist('prepareFoldRepresentation','file')==2,'Missing prepareFoldRepresentation.m');
assert(exist('materializeAlignedRFFP','file')==2,'Missing materializeAlignedRFFP.m');
if ~isfolder(outputDir),mkdir(outputDir);end
S=load(dataFile,'reps');assert(isfield(S,'reps'),'Missing reps variable');reps=S.reps;
assert(isfield(reps,'metadata') && istable(reps.metadata) && ...
 ismember('LogID',reps.metadata.Properties.VariableNames), ...
 'Expected reps.metadata table with LogID column.');
Y=double(reps.targetXY_m);N=size(Y,1);logID=double(reps.metadata.LogID(:));
assert(N==118930 && size(Y,2)==2 && all(isfinite(Y(:))),'Unexpected target schema');
assert(numel(logID)==N && all(ismember(logID,0:37)),'Unexpected LogID mapping');
year=2015*ones(N,1);year(ismember(logID,22:37))=2014;
assert(nnz(year==2014)==32798 && nnz(year==2015)==86132, ...
 'Campaign counts do not match verified mapping');
base=struct('MinTrainingObservations',50,'MaxLTEFeatures',128, ...
 'MaxWLANFeatures',384,'LTEMissingFloor_dBm',-140,'WLANMissingFloor_dBm',-105);
cfg=struct('seed',20261117,'elmWidths',[256 512 1024], ...
 'elmRidge',[1e-2 1e-1 1],'elmBatchSize',2048,'predBatchSize',2048, ...
 'validationGroups',5);
reports=struct([]);
for d=directions(:)'
 if d==1,trainYear=2014;testYear=2015;else,trainYear=2015;testYear=2014;end
 trainMask=year==trainYear;testMask=year==testYear;
 assert(~any(trainMask & testMask) && all(trainMask | testMask));
 assert(isempty(intersect(unique(logID(trainMask)),unique(logID(testMask)))), ...
 'Training/test LogID leakage');
 folder=fullfile(outputDir,sprintf('F%02d',d));if ~isfolder(folder),mkdir(folder);end
 completeFile=fullfile(folder,'completed_result.mat');
 if isfile(completeFile)
  Q=load(completeFile,'report');r=Q.report;
  fprintf('ELM direction %d already completed; not overwriting.\n',d);
  if isempty(reports),reports=r;else,reports(end+1)=r;end %#ok<AGROW>
  continue;
 end
 fprintf('\nELM cross-campaign %d -> %d | train %d | test %d\n', ...
 trainYear,testYear,nnz(trainMask),nnz(testMask));
 totalClock=tic;
 [fitMask,valMask,valLogs]=splitLogs(logID,trainMask,cfg.validationGroups,42+1000+d);
 assert(~any(fitMask & valMask) && isequal(fitMask|valMask,trainMask));
 assert(isempty(intersect(unique(logID(fitMask)),unique(logID(valMask)))));
 fprintf('Inner fit %d | validation %d | validation logs %d\n', ...
 nnz(fitMask),nnz(valMask),numel(valLogs));
 [XinFit,XinVal,innerFeatureInfo]=prepareFoldRepresentation(reps,"C",fitMask,valMask,base);
 [coordInner,ZinFit]=fitCoordinates(Y(fitMask,:));
 assert(all(isfinite(XinFit(:))) && all(isfinite(XinVal(:))));
 candidates=struct('Hidden',{},'Ridge',{},'ValMean_m',{},'ValP95_m',{},'Elapsed_s',{});
 k=0;
 for h=cfg.elmWidths
  rng(cfg.seed+200+h,'twister');D=size(XinFit,2);
  W=single(randn(D,h)/sqrt(double(D)));b=single(0.1*randn(1,h));
  clock=tic;[G,C]=elmMoments(XinFit,ZinFit,W,b,cfg.elmBatchSize);
  for lambda=cfg.elmRidge
   k=k+1;B=elmSolve(G,C,lambda);
   temp=struct('W',W,'b',b,'B',B);
   zhat=elmPredict(temp,XinVal,cfg.predBatchSize);
   pred=undoCoordinates(zhat,coordInner);
   met=calcMetrics(Y(valMask,:),pred);
   candidates(k)=struct('Hidden',h,'Ridge',lambda, ...
    'ValMean_m',met.Mean_m,'ValP95_m',met.P95_m,'Elapsed_s',toc(clock)); %#ok<AGROW>
   fprintf('  ELM h=%d lambda=%g | validation mean %.3f m | P95 %.3f m\n', ...
    h,lambda,met.Mean_m,met.P95_m);
  end
  clear W b G C B temp
 end
 [~,best]=min([candidates.ValMean_m]);chosen=candidates(best);
 fprintf('Selected h=%d lambda=%g (validation only)\n',chosen.Hidden,chosen.Ridge);
 save(fullfile(folder,'validation_selection.mat'), ...
 'candidates','chosen','innerFeatureInfo','valLogs','cfg','-v7.3');
 clear XinFit XinVal ZinFit
 [Xtrain,Xtest,featureInfo]=prepareFoldRepresentation(reps,"C",trainMask,testMask,base);
 [coordFinal,Ztrain]=fitCoordinates(Y(trainMask,:));
 assert(all(isfinite(Xtrain(:))) && all(isfinite(Xtest(:))));
 rng(cfg.seed+200+chosen.Hidden,'twister');D=size(Xtrain,2);h=chosen.Hidden;
 W=single(randn(D,h)/sqrt(double(D)));b=single(0.1*randn(1,h));
 t=tic;[G,C]=elmMoments(Xtrain,Ztrain,W,b,cfg.elmBatchSize);
 B=elmSolve(G,C,chosen.Ridge);fitSeconds=toc(t);
 elmFinal=struct('W',W,'b',b,'B',B,'lambda',chosen.Ridge,'hidden',h);
 t=tic;pred=undoCoordinates(elmPredict(elmFinal,Xtest,cfg.predBatchSize),coordFinal);
 inferenceSeconds=toc(t);met=calcMetrics(Y(testMask,:),pred);
 assert(met.Coverage==1,'ELM produced missing/nonfinite predictions');
 testRows=find(testMask);testLogID=logID(testMask);errors=met.Errors_m;
 report=struct('Direction',d,'TrainingYear',trainYear,'TestYear',testYear, ...
  'NumTrain',nnz(trainMask),'NumTest',nnz(testMask), ...
  'NumInnerFit',nnz(fitMask),'NumInnerVal',nnz(valMask), ...
  'InnerValidationLogIDs',valLogs,'Candidates',candidates, ...
  'ChosenCandidate',chosen,'Metrics',met,'Coverage',met.Coverage, ...
  'FitSeconds',fitSeconds,'InferenceSeconds',inferenceSeconds, ...
  'TotalSeconds',toc(totalClock),'Config',cfg,'TestIndices',testRows);
 save(fullfile(folder,'trained_model.mat'),'elmFinal','featureInfo','coordFinal','-v7.3');
 save(fullfile(folder,'predictions.mat'),'pred','errors','testRows','testLogID','-v7.3');
 save(completeFile,'report','-v7.3');
 fprintf('ELM %d -> %d | Mean %.3f | Median %.3f | RMSE %.3f | P68 %.3f | P95 %.3f | Coverage %.2f%% | total %.1f s\n', ...
 trainYear,testYear,met.Mean_m,met.Median_m,met.RMSE_m,met.P68_m,met.P95_m,100*met.Coverage,report.TotalSeconds);
 if isempty(reports),reports=report;else,reports(end+1)=report;end %#ok<AGROW>
 clear Xtrain Xtest Ztrain G C W b B elmFinal
end
end

function [fitMask,valMask,valLogs]=splitLogs(logID,trainMask,k,seed)
 u=unique(logID(trainMask));assert(numel(u)>=k,'Too few development logs');
 rng(seed,'twister');u=u(randperm(numel(u)));
 counts=arrayfun(@(id)nnz(trainMask & logID==id),u);
 [~,ord]=sort(counts,'descend');u=u(ord);counts=counts(ord);
 groupCount=zeros(k,1);assignment=zeros(numel(u),1);
 for j=1:numel(u)
  [~,g]=min(groupCount);assignment(j)=g;groupCount(g)=groupCount(g)+counts(j);
 end
 valLogs=u(assignment==1);
 valMask=trainMask & ismember(logID,valLogs);fitMask=trainMask & ~valMask;
 assert(any(fitMask)&&any(valMask),'Empty fit/validation split');
end
function [s,Z]=fitCoordinates(Y)
 s=struct('mu',mean(Y,1),'sigma',std(Y,0,1));
 s.sigma(~isfinite(s.sigma)|s.sigma==0)=1;
 Z=single((Y-s.mu)./s.sigma);
end
function Y=undoCoordinates(Z,s)
 Y=double(Z).*s.sigma+s.mu;
end
function [G,C]=elmMoments(X,Z,W,b,batchSize)
 H=size(W,2);G=zeros(H+1,H+1);C=zeros(H+1,2);
 for i=1:batchSize:size(X,1)
  ii=i:min(i+batchSize-1,size(X,1));
  A=double(max(0,single(X(ii,:))*W+b));
  A=[A,ones(numel(ii),1)];
  G=G+A'*A;C=C+A'*double(Z(ii,:));
 end
end
function B=elmSolve(G,C,lambda)
 R=eye(size(G));R(end,end)=0;
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
function m=calcMetrics(Y,Yhat)
 valid=all(isfinite(Yhat),2);
 e=hypot(Y(valid,1)-Yhat(valid,1),Y(valid,2)-Yhat(valid,2));
 assert(~isempty(e),'No valid predictions');
 m=struct('Mean_m',mean(e),'Median_m',median(e), ...
 'RMSE_m',sqrt(mean(e.^2)),'P68_m',prctile(e,68), ...
 'P95_m',prctile(e,95),'Max_m',max(e), ...
 'Coverage',nnz(valid)/size(Y,1),'Errors_m',e);
end
