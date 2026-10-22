function [fitMask,valMask,routeTable] = makeRFFPInnerValidation( ...
    metadata,outerTrainMask,numFolds,validationFold,seed)
%MAKERFFPINNERVALIDATION Create a log-grouped split within outer training.

arguments
    metadata table
    outerTrainMask (:,1) logical
    numFolds (1,1) double {mustBeInteger,mustBeGreaterThan(numFolds,1)}
    validationFold (1,1) double {mustBeInteger,mustBePositive}
    seed (1,1) double
end
assert(numel(outerTrainMask)==height(metadata));
assert(validationFold<=numFolds);

outerRows = find(outerTrainMask);
[innerIndex,routeTable] = makeRouteGroupedFolds( ...
    metadata(outerTrainMask,:),numFolds,seed);
fitMask = false(height(metadata),1);
valMask = false(height(metadata),1);
fitMask(outerRows(innerIndex~=validationFold)) = true;
valMask(outerRows(innerIndex==validationFold)) = true;

assert(~any(fitMask & valMask));
assert(all((fitMask | valMask)==outerTrainMask));
fitLogs = unique(metadata.LogID(fitMask));
valLogs = unique(metadata.LogID(valMask));
assert(isempty(intersect(fitLogs,valLogs)),'A route entered fit and validation.');
end

