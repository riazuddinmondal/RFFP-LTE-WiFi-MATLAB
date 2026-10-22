%% Fast, non-training verification for the corrected DNN package
clear; clc;
scriptDir=fileparts(mfilename('fullpath'));
projectDir=fileparts(scriptDir);
addpath(scriptDir,fullfile(projectDir,'rffp_parser'), ...
    fullfile(projectDir,'rffp_10fold'));
dataFile=fullfile(projectDir,'RFFP_corrected_representations.mat');
assert(isfile(dataFile),'Representation MAT file is missing.');
assert(exist('trainNetwork','file')==2, ...
    'Deep Learning Toolbox is required (trainNetwork not found).');
S=load(dataFile,'reps'); reps=S.reps;
[foldIndex,routeTable]=makeRouteGroupedFolds(reps.metadata,10,42);
assert(height(routeTable)==38);
trainMask=foldIndex~=1; testMask=foldIndex==1;
cfg=struct('MinTrainingObservations',50,'MaxLTEFeatures',128, ...
    'MaxWLANFeatures',384,'LTEMissingFloor_dBm',-140, ...
    'WLANMissingFloor_dBm',-105);
[XTrain,XTest,featureInfo]=prepareFoldRepresentation( ...
    reps,"C",trainMask,testMask,cfg);
assert(size(XTrain,1)==nnz(trainMask));
assert(size(XTest,1)==nnz(testMask));
assert(size(XTrain,2)==size(XTest,2));
assert(all(isfinite(XTrain),'all') && all(isfinite(XTest),'all'));
fprintf('Representation C: %d training x %d features; %d test samples.\n', ...
    size(XTrain,1),size(XTrain,2),size(XTest,1));
fprintf('Selected transmitters: %d. Detection masks included: %d.\n', ...
    height(featureInfo.RepresentationInfo), ...
    any(featureInfo.RepresentationInfo.IncludesDetectionMask));
if canUseGPU
    g=gpuDevice;
    fprintf('GPU ready: %s, %.2f GB available.\n',g.Name,g.AvailableMemory/1024^3);
else
    warning('RFFP:NoGPU','No supported GPU is currently available.');
end
fprintf('DNN setup verification passed.\n');

