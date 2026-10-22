%% Independent tuned DNN C Fold 1 — preserve public source
clear; clc;
root='C:\RFFP_Release_Validation';
softwareDir=fullfile(root,'software');
dataFile=fullfile(root,'private_data','RFFP_corrected_representations.mat');
referenceFile=fullfile(root,'reference','corrected_route10fold_checkpoint.mat');
outDir=fullfile(root,'test_outputs','DNN_Level3_Fold1');
assert(isfile(dataFile),'Missing private dataset: %s',dataFile);
assert(isfile(referenceFile),'Missing reference checkpoint: %s',referenceFile);
if ~isfolder(outDir), mkdir(outDir); end
addpath(genpath(softwareDir));
sourceFile=which('run_nested_tuned_RFFP_DNN_10fold');
assert(~isempty(sourceFile),'Tuned DNN runner not found.');
assert(startsWith(lower(sourceFile),lower(softwareDir)), 'Unexpected runner: %s',sourceFile);
source=fileread(sourceFile);
oldHeader=sprintf('scriptDir=fileparts(mfilename(''fullpath'')); projectDir=fileparts(scriptDir);\naddpath(scriptDir,fullfile(projectDir,''rffp_parser''), ...\n    fullfile(projectDir,''rffp_10fold''));');
assert(contains(source,oldHeader),'Runner header differs: review source before patching.');
source=strrep(source,oldHeader,sprintf('addpath(genpath(''%s''));',strrep(softwareDir,'\','/')));
oldData='dataFile=fullfile(projectDir,''RFFP_corrected_representations.mat'');';
oldOut='outputDir=fullfile(projectDir,''dnn_tuned_results'');';
assert(contains(source,oldData) && contains(source,oldOut),'Data/output declarations differ.');
source=strrep(source,oldData,sprintf('dataFile=''%s'';',strrep(dataFile,'\','/')));
source=strrep(source,oldOut,sprintf('outputDir=''%s'';',strrep(outDir,'\','/')));
assert(contains(source,'foldsToRun=1:10; numFolds=10; seed=42; rep="C";'), ...
    'Fold/seed declarations differ.');
source=strrep(source,'foldsToRun=1:10; numFolds=10; seed=42; rep="C";', ...
    'foldsToRun=1; numFolds=10; seed=42; rep="C";');
% Keep original candidate search, hyperparameters and training procedure.
assert(~isfile(fullfile(outDir,'nested_tuned_DNN_checkpoint.mat')), ...
    'Existing validation checkpoint found: move it before a fresh independent run.');
% Validate original partition; do not use the checkpoint for model resume.
R=load(referenceFile,'savedFoldIndex'); S=load(dataFile,'reps');
[f,~]=makeRouteGroupedFolds(S.reps.metadata,10,42);
assert(isequal(double(f(:)),double(R.savedFoldIndex(:))), 'Fold assignment differs.');
clear S R f
copyFile=fullfile(root,'test_outputs','run_DNN_Fold1_validation_copy.m');
fid=fopen(copyFile,'w'); assert(fid>0,'Cannot write validation copy.');
c=onCleanup(@() fclose(fid)); fwrite(fid,source,'char'); clear c
fprintf('DNN Fold 1 source copy: %s\nOutput: %s\n',copyFile,outDir);
fprintf('Full nested tuning + final training will run. This can take hours.\n');
cd(softwareDir); run(copyFile);
