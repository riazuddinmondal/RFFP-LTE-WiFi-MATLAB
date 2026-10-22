function [foldIndex, routeTable] = makeRouteGroupedFolds(metadata, numFolds, seed)
%MAKEROUTEGROUPEDFOLDS Assign complete Nemo logs to campaign-balanced folds.
% No LogID can occur in more than one fold.

arguments
    metadata table
    numFolds (1,1) double {mustBeInteger,mustBePositive} = 10
    seed (1,1) double = 42
end

rng(seed, 'twister');
logIDs = unique(metadata.LogID);
numRoutes = numel(logIDs);
routeCampaign = strings(numRoutes,1);
routeSamples = zeros(numRoutes,1);
routeFold = zeros(numRoutes,1);

for i = 1:numRoutes
    mask = metadata.LogID == logIDs(i);
    routeSamples(i) = nnz(mask);
    c = unique(string(metadata.Campaign(mask)));
    assert(numel(c) == 1, 'A LogID belongs to multiple campaigns.');
    routeCampaign(i) = c;
end

campaigns = unique(routeCampaign, 'stable');
totalFoldLoad = zeros(numFolds,1);
for c = 1:numel(campaigns)
    candidates = find(routeCampaign == campaigns(c));
    tieNoise = rand(numel(candidates),1);
    [~, order] = sortrows([-routeSamples(candidates), tieNoise], [1 2]);
    candidates = candidates(order);
    campaignLoad = zeros(numFolds,1);
    for j = 1:numel(candidates)
        best = find(campaignLoad == min(campaignLoad));
        [~, k] = min(totalFoldLoad(best));
        f = best(k);
        routeFold(candidates(j)) = f;
        campaignLoad(f) = campaignLoad(f) + routeSamples(candidates(j));
        totalFoldLoad(f) = totalFoldLoad(f) + routeSamples(candidates(j));
    end
end

foldIndex = zeros(height(metadata),1,'uint8');
for i = 1:numRoutes
    foldIndex(metadata.LogID == logIDs(i)) = uint8(routeFold(i));
end

assert(all(foldIndex >= 1 & foldIndex <= numFolds));
for i = 1:numRoutes
    assert(numel(unique(foldIndex(metadata.LogID == logIDs(i)))) == 1);
end

routeTable = table(logIDs, categorical(routeCampaign), routeSamples, routeFold, ...
    'VariableNames', {'LogID','Campaign','NumSamples','Fold'});
end
