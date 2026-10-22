function layers = buildRFFPMLP(numFeatures,cfg)
%BUILDRFFPMLP Compact fully connected network for aligned RF fingerprints.

arguments
    numFeatures (1,1) double {mustBeInteger,mustBePositive}
    cfg (1,1) struct
end
h = cfg.HiddenUnits(:)';
assert(numel(h)>=2,'HiddenUnits must contain at least two layer widths.');

layers = [
    featureInputLayer(numFeatures,'Normalization','none','Name','fingerprint')
];
for i = 1:numel(h)
    layers = [layers
        fullyConnectedLayer(h(i),'Name',sprintf('fc%d',i), ...
            'WeightsInitializer','he')
        batchNormalizationLayer('Name',sprintf('bn%d',i))
        reluLayer('Name',sprintf('relu%d',i))]; %#ok<AGROW>
    if i<=numel(cfg.Dropout) && cfg.Dropout(i)>0
        layers = [layers
            dropoutLayer(cfg.Dropout(i),'Name',sprintf('dropout%d',i))]; %#ok<AGROW>
    end
end
layers = [layers
    fullyConnectedLayer(2,'Name','normalized_EN')
    regressionLayer('Name','coordinate_MSE')];
end

