function summary = run_RFFP_Hierarchical_FastKAN_V2_10fold(dataFile,varargin)
%RUN_RFFP_HIERARCHICAL_FASTKAN_V2_10FOLD
% Checkpointed 10-fold wrapper for run_RFFP_Hierarchical_FastKAN_V2.m.
%
% This wrapper:
%   1) Runs the exact same V2 KAN configuration for every outer fold.
%   2) Saves each fold immediately (the V2 runner already writes its result
%      and model MAT files; this wrapper also updates a lightweight checkpoint).
%   3) Resumes automatically by skipping valid completed fold-result files.
%   4) Builds a final per-fold table and concatenated OUT-OF-FOLD prediction
%      result, with overall metrics calculated across all test samples.
%   5) Records per-fold wall-clock time, Global FastKAN time, local/hierarchical
%      KAN stage times, running average, and ETA after every completed fold.
%   6) Saves both MAT and CSV summary files.
%
% REQUIRED FILES IN MATLAB PATH/CURRENT FOLDER
%   run_RFFP_Hierarchical_FastKAN_V2.m
%   FastKANLayer.m
%   RFFP_corrected_representations.mat
%   corrected_route10fold_checkpoint.mat
%
% RECOMMENDED FULL RUN
%   summary = run_RFFP_Hierarchical_FastKAN_V2_10fold( ...
%       'RFFP_corrected_representations.mat', ...
%       'FoldFile','corrected_route10fold_checkpoint.mat', ...
%       'ExecutionEnvironment','auto');
%
% RESUME BEHAVIOUR
%   Resume=true is the default. If, for example, folds 1-4 already have
%   valid FastKAN_V2_fold#_results.mat files, the next run loads/skips them
%   and starts training at fold 5.
%
% START/END FOLD (useful only for controlled testing)
%   ...,'StartFold',3,'EndFold',10
%
% FORCE RE-RUN ALL SELECTED FOLDS
%   ...,'Resume',false
%
% TIMING COMPLETENESS
%   RequireTiming=true is the default. An older saved fold result that was
%   created before runtime instrumentation will be rerun so every fold in the
%   final timing table has comparable timing data. Set RequireTiming=false if
%   you prefer to reuse such older results even though their timing is unknown.
%
% Any V2 training name-value option can be passed here and is forwarded to
% run_RFFP_Hierarchical_FastKAN_V2, e.g. Epochs, BatchSize, NumGrids,
% CandidateFeatures, GlobalFeatures, LocalFeatures, MaxExperts, etc.
%
% IMPORTANT FOR THE JOURNAL EXPERIMENT:
% Keep the hyperparameters fixed across all 10 folds. Do not tune each fold
% individually using its test result.

%% -------------------------- wrapper options ----------------------------
if nargin < 1 || isempty(dataFile)
    dataFile = 'RFFP_corrected_representations.mat';
end
if ~isfile(dataFile)
    error('Data file not found: %s',dataFile);
end
if ~isfile('run_RFFP_Hierarchical_FastKAN_V2.m') && ...
        exist('run_RFFP_Hierarchical_FastKAN_V2','file') ~= 2
    error('run_RFFP_Hierarchical_FastKAN_V2.m is not on the MATLAB path.');
end
if ~isfile('FastKANLayer.m') && exist('FastKANLayer','class') ~= 8
    error('FastKANLayer.m is not on the MATLAB path.');
end

% Wrapper-only parameters are removed before forwarding to the V2 runner.
resume        = getNV(varargin,'Resume',true);
stopOnError   = getNV(varargin,'StopOnError',true);
requireTiming = getNV(varargin,'RequireTiming',true);
startFold     = getNV(varargin,'StartFold',1);
endFold     = getNV(varargin,'EndFold',10);

validateattributes(resume,{'logical','numeric'},{'scalar'});
validateattributes(stopOnError,{'logical','numeric'},{'scalar'});
validateattributes(requireTiming,{'logical','numeric'},{'scalar'});
validateattributes(startFold,{'numeric'},{'scalar','integer','>=',1,'<=',10});
validateattributes(endFold,{'numeric'},{'scalar','integer','>=',1,'<=',10});
if startFold > endFold
    error('StartFold must be <= EndFold.');
end
resume = logical(resume);
stopOnError = logical(stopOnError);
requireTiming = logical(requireTiming);

foldFile  = string(getNV(varargin,'FoldFile','corrected_route10fold_checkpoint.mat'));
outputDir = string(getNV(varargin,'OutputDir','KAN_MATLAB_results'));
if ~isfile(foldFile)
    error('Fold file not found: %s',foldFile);
end
if ~isfolder(outputDir), mkdir(outputDir); end

% Remove wrapper-only pairs. Keep all V2 options, including FoldFile and
% OutputDir. Add FoldFile/OutputDir if the caller omitted them.
forwardArgs = removeNV(varargin,{'Resume','StopOnError','RequireTiming','StartFold','EndFold'});
if ~hasNV(forwardArgs,'FoldFile')
    forwardArgs = [forwardArgs {'FoldFile',char(foldFile)}]; %#ok<AGROW>
end
if ~hasNV(forwardArgs,'OutputDir')
    forwardArgs = [forwardArgs {'OutputDir',char(outputDir)}]; %#ok<AGROW>
end

checkpointFile = fullfile(outputDir,'FastKAN_V2_10fold_checkpoint.mat');
summaryMatFile = fullfile(outputDir,'FastKAN_V2_10fold_combined_results.mat');
summaryCsvFile = fullfile(outputDir,'FastKAN_V2_10fold_fold_metrics.csv');

fprintf('\n============================================================\n');
fprintf('Hierarchical LTE/Wi-Fi FastKAN RFFP V2 - 10-FOLD RUNNER\n');
fprintf('Data:       %s\n',dataFile);
fprintf('Fold file:  %s\n',foldFile);
fprintf('Output dir: %s\n',outputDir);
fprintf('Folds:      %d to %d\n',startFold,endFold);
fprintf('Resume:     %s\n',string(resume));
fprintf('Require timing on resumed folds: %s\n',string(requireTiming));
fprintf('============================================================\n');

%% ----------------------------- run folds -------------------------------
foldResults = cell(10,1);
completed = false(10,1);
failed = false(10,1);
failureMessage = strings(10,1);
foldResultFiles = strings(10,1);
foldModelFiles = strings(10,1);
foldWallClock_s = nan(10,1);

% First discover any already-saved results. This also lets a restarted run
% create a complete summary without retraining prior folds.
for f = 1:10
    rf = fullfile(outputDir,sprintf('FastKAN_V2_fold%d_results.mat',f));
    mf = fullfile(outputDir,sprintf('FastKAN_V2_fold%d_models.mat',f));
    foldResultFiles(f) = string(rf);
    foldModelFiles(f) = string(mf);
    if resume && isfile(rf)
        try
            T = load(rf,'results');
            if isfield(T,'results') && isfield(T.results,'Fold') && T.results.Fold == f && ...
                    isfield(T.results,'TestIndex') && isfield(T.results,'PositionError_m')
                hasCoreTiming = isfield(T.results,'Timing') && isstruct(T.results.Timing) && ...
                    isfield(T.results.Timing,'AlgorithmTotal_s');
                if requireTiming && ~hasCoreTiming
                    fprintf('Fold %d saved result has no detailed runtime record -> will RERUN for timing.\n',f);
                else
                    foldResults{f} = T.results;
                    completed(f) = true;
                    if isfield(T.results,'Timing') && isfield(T.results.Timing,'FoldWallClock_s')
                        foldWallClock_s(f) = T.results.Timing.FoldWallClock_s;
                    elseif hasCoreTiming
                        foldWallClock_s(f) = T.results.Timing.AlgorithmTotal_s;
                    end
                end
            end
        catch ME
            warning('Fold %d saved result could not be loaded and will be rerun: %s',f,ME.message);
        end
    end
end

saveCheckpoint(checkpointFile,completed,failed,failureMessage, ...
    foldResultFiles,foldModelFiles,foldWallClock_s,startFold,endFold);

for f = startFold:endFold
    fprintf('\n************************************************************\n');
    fprintf('10-fold KAN: OUTER FOLD %d / 10\n',f);
    fprintf('************************************************************\n');

    if resume && completed(f)
        m = foldResults{f}.HierarchicalMetrics;
        fprintf(['Fold %d already completed -> SKIPPED. ' ...
            'Mean %.2f | Median %.2f | P68 %.2f | P95 %.2f | N %d\n'], ...
            f,m.MeanPE_m,m.MedianPE_m,m.P68_m,m.P95_m,m.N);
        if isfinite(foldWallClock_s(f))
            fprintf('  Saved fold runtime: %s\n',formatDuration(foldWallClock_s(f)));
        end
        continue;
    end

    tFold = tic;
    try
        r = run_RFFP_Hierarchical_FastKAN_V2(dataFile,f,forwardArgs{:});
        elapsed_s = toc(tFold);
        foldWallClock_s(f) = elapsed_s;
        if ~isfield(r,'Timing') || ~isstruct(r.Timing), r.Timing = struct; end
        r.Timing.FoldWallClock_s = elapsed_s;
        r.Timing.FoldCompletedAt = datetime('now');

        foldResults{f} = r;
        completed(f) = true;
        failed(f) = false;
        failureMessage(f) = "";

        % The V2 function already saved the full result and model files.
        % Save the returned result again with wrapper wall-clock timing.
        rf = char(foldResultFiles(f));
        results = r; %#ok<NASGU>
        save(rf,'results','-v7.3');

        m = r.HierarchicalMetrics;
        fprintf(['\nFOLD %d COMPLETE | wall clock %s | Mean %.2f | Median %.2f | ' ...
            'RMSE %.2f | P68 %.2f | P95 %.2f | N %d\n'], ...
            f,formatDuration(elapsed_s),m.MeanPE_m,m.MedianPE_m,m.RMSE_m,m.P68_m,m.P95_m,m.N);
        if isfield(r,'Timing')
            fprintf('  Global FastKAN       : %s\n',formatDuration(getTiming(r,'GlobalFastKAN_s')));
            fprintf('  Local FastKAN train  : %s\n',formatDuration(getTiming(r,'LocalFastKANTraining_s')));
            fprintf('  Hierarchical stage   : %s\n',formatDuration(getTiming(r,'HierarchicalStage_s')));
            fprintf('  KAN algorithm total  : %s\n',formatDuration(getTiming(r,'AlgorithmTotal_s')));
        end

        doneTimes = foldWallClock_s(completed & isfinite(foldWallClock_s));
        avgFold_s = mean(doneTimes,'omitnan');
        remainingRequested = nnz(~completed(startFold:endFold));
        fprintf('  Running average/fold : %s\n',formatDuration(avgFold_s));
        fprintf('  Estimated remaining  : %s (%d fold%s)\n', ...
            formatDuration(avgFold_s*remainingRequested),remainingRequested,pluralS(remainingRequested));

        saveCheckpoint(checkpointFile,completed,failed,failureMessage, ...
            foldResultFiles,foldModelFiles,foldWallClock_s,startFold,endFold);

        % Write a running CSV after every completed fold so progress can be
        % inspected even while the 10-fold experiment is unfinished.
        partialTable = buildFoldTable(foldResults,completed,failed,failureMessage,foldWallClock_s);
        writetable(partialTable,summaryCsvFile);

    catch ME
        failed(f) = true;
        failureMessage(f) = string(getReport(ME,'extended','hyperlinks','off'));
        warning('Fold %d failed: %s',f,ME.message);
        saveCheckpoint(checkpointFile,completed,failed,failureMessage, ...
            foldResultFiles,foldModelFiles,foldWallClock_s,startFold,endFold);
        partialTable = buildFoldTable(foldResults,completed,failed,failureMessage,foldWallClock_s);
        writetable(partialTable,summaryCsvFile);
        if stopOnError
            rethrow(ME);
        end
    end
end

%% --------------------- reload all available folds ----------------------
% Reload from disk rather than relying only on RAM. This makes the combined
% result robust to resumed runs and keeps each fold independently auditable.
for f = 1:10
    rf = char(foldResultFiles(f));
    if isfile(rf)
        try
            T = load(rf,'results');
            if isfield(T,'results') && isfield(T.results,'Fold') && T.results.Fold == f
                foldResults{f} = T.results;
                completed(f) = true;
                if isfield(T.results,'Timing') && isfield(T.results.Timing,'FoldWallClock_s')
                    foldWallClock_s(f) = T.results.Timing.FoldWallClock_s;
                elseif isfield(T.results,'Timing') && isfield(T.results.Timing,'AlgorithmTotal_s')
                    foldWallClock_s(f) = T.results.Timing.AlgorithmTotal_s;
                end
            end
        catch ME
            warning('Could not reload fold %d result for final aggregation: %s',f,ME.message);
        end
    end
end

foldTable = buildFoldTable(foldResults,completed,failed,failureMessage,foldWallClock_s);
writetable(foldTable,summaryCsvFile);

%% ---------------- concatenate out-of-fold predictions -----------------
validFolds = find(completed);
if isempty(validFolds)
    error('No successfully completed fold result is available.');
end

allIndex = [];
allTrue = [];
allPred = [];
allGlobalPred = [];
allError = [];
allUsedLocal = [];
allFold = [];

for k = 1:numel(validFolds)
    f = validFolds(k);
    r = foldResults{f};
    n = numel(r.TestIndex);
    allIndex = [allIndex; double(r.TestIndex(:))]; %#ok<AGROW>
    allTrue = [allTrue; single(r.TrueXY_m)]; %#ok<AGROW>
    allPred = [allPred; single(r.PredictedXY_m)]; %#ok<AGROW>
    allGlobalPred = [allGlobalPred; single(r.GlobalPredictedXY_m)]; %#ok<AGROW>
    allError = [allError; double(r.PositionError_m(:))]; %#ok<AGROW>
    allUsedLocal = [allUsedLocal; logical(r.UsedLocalExpert(:))]; %#ok<AGROW>
    allFold = [allFold; repmat(f,n,1)]; %#ok<AGROW>
end

% Sort by original sample index so every row corresponds directly to the
% original corrected representation.
[allIndex,ord] = sort(allIndex);
allTrue = allTrue(ord,:);
allPred = allPred(ord,:);
allGlobalPred = allGlobalPred(ord,:);
allError = allError(ord);
allUsedLocal = allUsedLocal(ord);
allFold = allFold(ord);

if numel(unique(allIndex)) ~= numel(allIndex)
    error('Duplicate TestIndex values detected across completed outer folds. Check the fold files/results.');
end

% Recompute errors from saved coordinates as an integrity check.
recalcError = sqrt(sum((double(allPred)-double(allTrue)).^2,2));
maxErrorMismatch = max(abs(recalcError-allError));
if maxErrorMismatch > 1e-3
    warning('Saved PositionError_m differs from recomputed coordinate error (max mismatch %.6g m). Using recomputed values.',maxErrorMismatch);
    allError = recalcError;
end

overallMetrics = metricsFromError(allError);
globalError = sqrt(sum((double(allGlobalPred)-double(allTrue)).^2,2));
overallGlobalMetrics = metricsFromError(globalError);
overallLocalCoverage = mean(allUsedLocal);

combined = struct;
combined.Method = 'Hierarchical LTE-WiFi FastKAN V2 - route-grouped out-of-fold';
combined.CompletedFolds = validFolds(:).';
combined.IsComplete10Fold = all(completed);
combined.N = numel(allIndex);
combined.TestIndex = allIndex;
combined.Fold = allFold;
combined.TrueXY_m = allTrue;
combined.PredictedXY_m = allPred;
combined.GlobalPredictedXY_m = allGlobalPred;
combined.PositionError_m = allError;
combined.GlobalPositionError_m = globalError;
combined.UsedLocalExpert = allUsedLocal;
combined.LocalCoverage = overallLocalCoverage;
combined.OverallMetrics = overallMetrics;
combined.OverallGlobalMetrics = overallGlobalMetrics;
combined.FoldMetrics = foldTable;
combined.FoldResultFiles = foldResultFiles;
combined.FoldModelFiles = foldModelFiles;
combined.FoldWallClock_s = foldWallClock_s;
combined.TotalCompletedFoldWallClock_s = sum(foldWallClock_s(completed),'omitnan');
combined.MeanCompletedFoldWallClock_s = mean(foldWallClock_s(completed),'omitnan');
combined.DataFile = string(dataFile);
combined.FoldFile = foldFile;
combined.Created = datetime('now');

save(summaryMatFile,'combined','-v7.3');
saveCheckpoint(checkpointFile,completed,failed,failureMessage, ...
    foldResultFiles,foldModelFiles,foldWallClock_s,startFold,endFold);

%% ------------------------------- report --------------------------------
fprintf('\n============================================================\n');
fprintf('FASTKAN V2 OUT-OF-FOLD SUMMARY\n');
fprintf('Completed folds: %s\n',mat2str(validFolds(:).'));
fprintf('Total OOF samples: %d\n',combined.N);
fprintf('============================================================\n');
disp(foldTable(:,{'Fold','Status','N','MeanPE_m','MedianPE_m','RMSE_m','P68_m','P95_m','MaxPE_m', ...
    'GlobalFastKAN_min','LocalFastKANTrain_min','HierarchicalStage_min','AlgorithmTotal_min','FoldWallClock_min','LocalCoverage'}));

fprintf('\nOVERALL Hierarchical FastKAN V2 (concatenated OOF predictions):\n');
fprintf('  Mean %.2f | Median %.2f | RMSE %.2f | P68 %.2f | P95 %.2f | Max %.2f | N %d\n', ...
    overallMetrics.MeanPE_m,overallMetrics.MedianPE_m,overallMetrics.RMSE_m, ...
    overallMetrics.P68_m,overallMetrics.P95_m,overallMetrics.MaxPE_m,overallMetrics.N);
fprintf('  Accepted-expert OOF coverage: %.2f %%\n',100*overallLocalCoverage);

fprintf('\nOVERALL Global FastKAN only (same OOF samples):\n');
fprintf('  Mean %.2f | Median %.2f | RMSE %.2f | P68 %.2f | P95 %.2f | Max %.2f | N %d\n', ...
    overallGlobalMetrics.MeanPE_m,overallGlobalMetrics.MedianPE_m,overallGlobalMetrics.RMSE_m, ...
    overallGlobalMetrics.P68_m,overallGlobalMetrics.P95_m,overallGlobalMetrics.MaxPE_m,overallGlobalMetrics.N);

validTiming = completed & isfinite(foldWallClock_s);
if any(validTiming)
    fprintf('\nRUNTIME SUMMARY (%d completed folds with timing):\n',nnz(validTiming));
    fprintf('  Mean wall-clock per fold : %s\n',formatDuration(mean(foldWallClock_s(validTiming),'omitnan')));
    fprintf('  Total recorded wall time : %s\n',formatDuration(sum(foldWallClock_s(validTiming),'omitnan')));
end

if all(completed)
    fprintf('\n10/10 folds complete. Final combined result is ready for model comparison.\n');
else
    missing = find(~completed);
    fprintf('\nWARNING: this is a PARTIAL OOF summary. Missing folds: %s\n',mat2str(missing(:).'));
    fprintf('Run this same command again with Resume=true to continue.\n');
end
fprintf('\nSaved fold summary CSV:\n  %s\n',summaryCsvFile);
fprintf('Saved combined OOF MAT:\n  %s\n',summaryMatFile);
fprintf('Saved resume checkpoint:\n  %s\n',checkpointFile);

summary = combined;
end

%% =======================================================================
function value = getNV(args,name,defaultValue)
value = defaultValue;
for i = 1:2:numel(args)
    if i+1 <= numel(args) && (ischar(args{i}) || isstring(args{i})) && strcmpi(string(args{i}),string(name))
        value = args{i+1};
        return;
    end
end
end

function tf = hasNV(args,name)
tf = false;
for i = 1:2:numel(args)
    if i <= numel(args) && (ischar(args{i}) || isstring(args{i})) && strcmpi(string(args{i}),string(name))
        tf = true;
        return;
    end
end
end

function out = removeNV(args,names)
out = {};
i = 1;
while i <= numel(args)
    if i == numel(args)
        error('Name-value arguments must occur in pairs. Unpaired argument: %s',string(args{i}));
    end
    nm = args{i};
    if ~(ischar(nm) || isstring(nm))
        error('Expected a name-value parameter name at argument position %d.',i);
    end
    if ~any(strcmpi(string(nm),string(names)))
        out(end+1:end+2) = args(i:i+1); %#ok<AGROW>
    end
    i = i + 2;
end
end

function saveCheckpoint(filename,completed,failed,failureMessage,foldResultFiles,foldModelFiles,foldWallClock_s,startFold,endFold)
checkpoint = struct;
checkpoint.Completed = completed;
checkpoint.Failed = failed;
checkpoint.FailureMessage = failureMessage;
checkpoint.CompletedFolds = find(completed).';
checkpoint.FailedFolds = find(failed).';
checkpoint.FoldResultFiles = foldResultFiles;
checkpoint.FoldModelFiles = foldModelFiles;
checkpoint.FoldWallClock_s = foldWallClock_s;
checkpoint.RequestedFoldRange = [startFold endFold];
checkpoint.LastUpdated = datetime('now');
save(filename,'checkpoint');
end

function T = buildFoldTable(foldResults,completed,failed,failureMessage,foldWallClock_s)
Fold = (1:10).';
Status = repmat("pending",10,1);
N = nan(10,1);
MeanPE_m = nan(10,1);
MedianPE_m = nan(10,1);
RMSE_m = nan(10,1);
P68_m = nan(10,1);
P95_m = nan(10,1);
MaxPE_m = nan(10,1);
LocalCoverage = nan(10,1);
MetaFusionAlpha = nan(10,1);
MetaFusionImprovement = nan(10,1);
GlobalTraining_min = nan(10,1);
GlobalInference_min = nan(10,1);
GlobalFastKAN_min = nan(10,1);
LocalFastKANTrain_min = nan(10,1);
LocalExpertsStage_min = nan(10,1);
FusionCalibration_min = nan(10,1);
HierarchicalInference_min = nan(10,1);
HierarchicalStage_min = nan(10,1);
AlgorithmTotal_min = nan(10,1);
FoldWallClock_min = nan(10,1);

for f = 1:10
    if completed(f) && ~isempty(foldResults{f})
        r = foldResults{f};
        m = r.HierarchicalMetrics;
        Status(f) = "complete";
        N(f) = m.N;
        MeanPE_m(f) = m.MeanPE_m;
        MedianPE_m(f) = m.MedianPE_m;
        RMSE_m(f) = m.RMSE_m;
        P68_m(f) = m.P68_m;
        P95_m(f) = m.P95_m;
        MaxPE_m(f) = m.MaxPE_m;
        LocalCoverage(f) = r.LocalCoverage;
        if isfield(r,'MetaFusionAlpha'), MetaFusionAlpha(f) = r.MetaFusionAlpha; end
        if isfield(r,'MetaFusionImprovement'), MetaFusionImprovement(f) = r.MetaFusionImprovement; end
        if isfield(r,'Timing')
            GlobalTraining_min(f) = getTiming(r,'GlobalTraining_s')/60;
            GlobalInference_min(f) = getTiming(r,'GlobalInference_s')/60;
            GlobalFastKAN_min(f) = getTiming(r,'GlobalFastKAN_s')/60;
            LocalFastKANTrain_min(f) = getTiming(r,'LocalFastKANTraining_s')/60;
            LocalExpertsStage_min(f) = getTiming(r,'LocalExpertsStage_s')/60;
            FusionCalibration_min(f) = getTiming(r,'FusionCalibration_s')/60;
            HierarchicalInference_min(f) = getTiming(r,'HierarchicalInference_s')/60;
            HierarchicalStage_min(f) = getTiming(r,'HierarchicalStage_s')/60;
            AlgorithmTotal_min(f) = getTiming(r,'AlgorithmTotal_s')/60;
        end
        if isfinite(foldWallClock_s(f)), FoldWallClock_min(f) = foldWallClock_s(f)/60; end
    elseif failed(f)
        Status(f) = "failed";
    end
end

Failure = strings(10,1);
for f=1:10
    if failed(f), Failure(f) = failureMessage(f); end
end

T = table(Fold,Status,N,MeanPE_m,MedianPE_m,RMSE_m,P68_m,P95_m,MaxPE_m, ...
    LocalCoverage,MetaFusionAlpha,MetaFusionImprovement, ...
    GlobalTraining_min,GlobalInference_min,GlobalFastKAN_min,LocalFastKANTrain_min, ...
    LocalExpertsStage_min,FusionCalibration_min,HierarchicalInference_min, ...
    HierarchicalStage_min,AlgorithmTotal_min,FoldWallClock_min,Failure);
end

function value = getTiming(r,fieldName)
value = NaN;
if isfield(r,'Timing') && isstruct(r.Timing) && isfield(r.Timing,fieldName)
    value = double(r.Timing.(fieldName));
end
end

function txt = formatDuration(secondsValue)
secondsValue = double(secondsValue);
if isempty(secondsValue) || ~isfinite(secondsValue)
    txt = 'unknown';
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

function s = pluralS(n)
if n == 1, s = ''; else, s = 's'; end
end

function m = metricsFromError(e)
e = double(e(:));
e = e(isfinite(e));
if isempty(e)
    m = struct('MeanPE_m',NaN,'MedianPE_m',NaN,'RMSE_m',NaN, ...
        'P68_m',NaN,'P95_m',NaN,'MaxPE_m',NaN,'N',0);
    return;
end
m = struct;
m.MeanPE_m = mean(e);
m.MedianPE_m = median(e);
m.RMSE_m = sqrt(mean(e.^2));
m.P68_m = prctile(e,68);
m.P95_m = prctile(e,95);
m.MaxPE_m = max(e);
m.N = numel(e);
end
