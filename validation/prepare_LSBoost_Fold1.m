%% RFFP Level 3 LSBoost Fold 1: prepare and launch original runner
% Run from a clean copy of the PUBLIC software folder. Do not overwrite the official runner or any original checkpoint.
clear; clc;
root = 'C:\RFFP_Release_Validation';
softwareDir = fullfile(root,'software');
privateDataFile = fullfile(root,'private_data','RFFP_corrected_representations.mat');
referenceFile = fullfile(root,'reference','corrected_route10fold_checkpoint.mat');
outputDir = fullfile(root,'test_outputs');
if ~isfolder(outputDir), mkdir(outputDir); end
assert(isfolder(softwareDir),'Software folder missing: %s',softwareDir);
assert(isfile(privateDataFile),'Dataset missing: %s',privateDataFile);
assert(isfile(referenceFile),'Reference checkpoint missing: %s',referenceFile);
runner = which('run_RFFP_10fold');
addpath(genpath(softwareDir));
runner = which('run_RFFP_10fold');
assert(~isempty(runner),'run_RFFP_10fold.m not found on MATLAB path');
assert(startsWith(lower(runner),lower(softwareDir)), 'Unexpected runner location: %s',runner);
% The runner is a script, so create a separate, modified validation copy.
source = fileread(runner);
assert(contains(source,"dataFile='RFFP_corrected_representations.mat';"), ...
    'Runner dataFile declaration differs; inspect runner manually.');
assert(contains(source,"checkpointFile='corrected_route10fold_checkpoint.mat';"), ...
    'Runner checkpoint declaration differs; inspect runner manually.');
assert(contains(source,'representations=["A","B","C"];'), ...
    'Runner representation declaration differs; inspect manually.');
assert(contains(source,'methods=["LSBoost","KNN","AHC","FCM"];'), ...
    'Runner methods declaration differs; inspect manually.');
assert(contains(source,'foldsToRun=1:numFolds;'), ...
    'Runner folds declaration differs; inspect manually.');
assert(contains(source,'requestedWorkers=2;'), ...
    'Runner worker declaration differs; inspect manually.');
% Use forward slashes, accepted by MATLAB on Windows, for reliable quoting.
dataPath = strrep(privateDataFile,'\','/');
checkpointPath = strrep(fullfile(outputDir,'RFFP_Level3_LSBoost_Fold1.mat'),'\','/');
source = strrep(source,"dataFile='RFFP_corrected_representations.mat';", ...
    sprintf("dataFile='%s';",dataPath));
source = strrep(source,"checkpointFile='corrected_route10fold_checkpoint.mat';", ...
    sprintf("checkpointFile='%s';",checkpointPath));
source = strrep(source,'representations=["A","B","C"];','representations="C";');
source = strrep(source,'methods=["LSBoost","KNN","AHC","FCM"];','methods="LSBoost";');
source = strrep(source,'foldsToRun=1:numFolds;','foldsToRun=1;');
% Keep two-worker setting on department PC; original runner falls back
% to serial if parallel pool is unavailable.
validationRunner = fullfile(outputDir,'run_RFFP_LSBoost_Fold1_validation.m');
fid = fopen(validationRunner,'w');
assert(fid>0,'Cannot create validation runner');
cleanup = onCleanup(@() fclose(fid));
fwrite(fid,source,'char');
clear cleanup
fprintf('Validation runner created: %s\n',validationRunner);
fprintf('Original runner unchanged: %s\n',runner);
fprintf('Input dataset: %s\n',privateDataFile);
fprintf('New checkpoint: %s\n',checkpointPath);
fprintf('Reference checkpoint will not be modified.\n');
fprintf('\nLaunching LSBoost C Fold 1. This may take hours.\n');
cd(softwareDir);
run(validationRunner);
