function check_release(projectRoot)
% Structural checks before a clean MATLAB smoke test; not a training run.
if nargin<1,projectRoot=fileparts(fileparts(mfilename('fullpath')));end
addpath(projectRoot,fullfile(projectRoot,'rffp_parser'), ...
 fullfile(projectRoot,'rffp_10fold'),fullfile(projectRoot,'rffp_dnn'));
required={'materializeAlignedRFFP','prepareFoldRepresentation','makeRouteGroupedFolds', ...
 'positioningMetrics','trainLSBoostXY2Workers','buildRFFPMLP', ...
 'makeRFFPDNNCandidates','evaluateRFFPDNNCandidate', ...
 'trainFinalRFFPDNN','FastKANLayer','run_RFFP_Hierarchical_FastKAN_V2'};
for j=1:numel(required),assert(~isempty(which(required{j})), 'Missing dependency: %s',required{j});end
input=fullfile(projectRoot,'RFFP_corrected_representations.mat');
assert(isfile(input),'Place the reviewed processed MAT at the project root.');
S=load(input,'reps');N=size(S.reps.targetXY_m,1);assert(N==118930);
[fold,~]=makeRouteGroupedFolds(S.reps.metadata,10,42);
F=load(fullfile(projectRoot,'reference','corrected_route10fold_checkpoint.mat'),'savedFoldIndex');
assert(isequal(fold,F.savedFoldIndex),'Recomputed outer folds differ.');
fprintf('Structural checks passed; this does not train or validate the models.\n');
end
