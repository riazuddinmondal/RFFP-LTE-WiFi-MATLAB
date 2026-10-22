function [mx,my] = trainLSBoostXY2Workers(XTrain,Yn,cfg,useParallel,baseSeed)
%TRAINLSBOOSTXY2WORKERS Train independent X/Y LSBoost regressors on 2 workers.
%
%   [mx,my] = trainLSBoostXY2Workers(XTrain,Yn,cfg,useParallel,baseSeed)
%   trains the two coordinate regressors concurrently when useParallel is
%   true. The caller is responsible for creating a pool with two workers.
%
%   If parallel execution is disabled, the same helper falls back to
%   sequential training. Hyperparameters and training samples are unchanged.

arguments
    XTrain {mustBeNumeric}
    Yn {mustBeNumeric}
    cfg (1,1) struct
    useParallel (1,1) logical = false
    baseSeed (1,1) double = 42
end

assert(size(Yn,2)==2,'Yn must contain exactly two target coordinates.');

if useParallel
    models = cell(1,2);
    parfor c = 1:2
        % Give each coordinate model a fixed stream so repeated runs are
        % reproducible regardless of worker scheduling order.
        rng(baseSeed + c,'twister');
        treeTemplate = templateTree('MaxNumSplits',cfg.MaxNumSplits);
        models{c} = fitrensemble(XTrain,Yn(:,c), ...
            'Method','LSBoost', ...
            'NumLearningCycles',cfg.NumTrees, ...
            'Learners',treeTemplate, ...
            'LearnRate',cfg.LearnRate);
    end
    mx = models{1};
    my = models{2};
else
    % Serial fallback for machines without Parallel Computing Toolbox.
    rng(baseSeed,'twister');
    treeTemplate = templateTree('MaxNumSplits',cfg.MaxNumSplits);
    mx = fitrensemble(XTrain,Yn(:,1), ...
        'Method','LSBoost', ...
        'NumLearningCycles',cfg.NumTrees, ...
        'Learners',treeTemplate, ...
        'LearnRate',cfg.LearnRate);
    my = fitrensemble(XTrain,Yn(:,2), ...
        'Method','LSBoost', ...
        'NumLearningCycles',cfg.NumTrees, ...
        'Learners',treeTemplate, ...
        'LearnRate',cfg.LearnRate);
end
end
