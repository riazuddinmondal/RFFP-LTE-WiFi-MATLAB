function [X, featureInfo] = materializeAlignedRFFP(aligned, trainIdx, rows, opts)
%MATERIALIZEALIGNEDRFFP Create dense aligned features without test leakage.
%   [X, INFO] = MATERIALIZEALIGNEDRFFP(ALIGNED, TRAINIDX, ROWS) selects
%   transmitters solely by occurrence in TRAINIDX, then materializes ROWS.
%   Call it with the same TRAINIDX for training, validation and test rows.

arguments
    aligned (1,1) struct
    trainIdx
    rows
    opts.MinTrainingObservations (1,1) double {mustBeInteger,mustBePositive} = 50
    opts.MaxLTEFeatures (1,1) double {mustBeInteger,mustBeNonnegative} = 128
    opts.MaxWLANFeatures (1,1) double {mustBeInteger,mustBeNonnegative} = 384
    opts.LTEMissingFloor_dBm (1,1) double = -140
    opts.WLANMissingFloor_dBm (1,1) double = -105
    opts.IncludeDetectionMask (1,1) logical = true
end

P = aligned.PowerObserved_dBm;
n = size(P,1);
trainIdx = normalizeRows(trainIdx, n, 'trainIdx');
rows = normalizeRows(rows, n, 'rows');

occurrence = full(sum(spones(P(trainIdx,:)), 1));
isLTE = aligned.Technology == 'LTE';
eligible = occurrence >= opts.MinTrainingObservations;

lteCols = chooseMostFrequent(find(eligible & isLTE'), occurrence, opts.MaxLTEFeatures);
wlanCols = chooseMostFrequent(find(eligible & ~isLTE'), occurrence, opts.MaxWLANFeatures);
selected = [lteCols(:); wlanCols(:)];

numRows = numel(rows);
floors = [repmat(opts.LTEMissingFloor_dBm, numel(lteCols), 1); ...
          repmat(opts.WLANMissingFloor_dBm, numel(wlanCols), 1)];
Xpower = repmat(single(floors'), numRows, 1);
sub = P(rows, selected);
[rr, cc, vv] = find(sub);
Xpower(sub2ind(size(Xpower), rr, cc)) = single(vv);

if opts.IncludeDetectionMask
    X = [Xpower, single(full(spones(sub)))];
else
    X = Xpower;
end

featureInfo = table( ...
    aligned.TransmitterID(selected), aligned.Technology(selected), ...
    occurrence(selected)', floors, ...
    'VariableNames', {'TransmitterID','Technology', ...
    'TrainingOccurrences','MissingFloor_dBm'});
featureInfo.SelectedAlignedColumn = selected;
featureInfo.IncludesDetectionMask = repmat(opts.IncludeDetectionMask, height(featureInfo), 1);
end

function idx = normalizeRows(idx, n, name)
if islogical(idx)
    if numel(idx) ~= n
        error('RFFP:BadLogicalIndex', '%s must have %d elements.', name, n);
    end
    idx = find(idx);
else
    idx = idx(:);
    if any(idx < 1 | idx > n | idx ~= floor(idx))
        error('RFFP:BadRowIndex', '%s contains invalid row indices.', name);
    end
end
end

function selected = chooseMostFrequent(candidates, occurrence, maxCount)
if isempty(candidates) || maxCount == 0
    selected = zeros(0,1);
    return;
end
[~, order] = sort(occurrence(candidates), 'descend');
selected = candidates(order(1:min(maxCount, numel(order))));
end
