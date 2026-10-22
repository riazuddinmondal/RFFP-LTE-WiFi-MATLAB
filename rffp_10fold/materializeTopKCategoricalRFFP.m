function [X, info] = materializeTopKCategoricalRFFP(topK, trainIdx, rows, opts)
%MATERIALIZETOPKCATEGORICALRFFP Exact one-hot identity encoding of top-K IDs.
% The transmitter dictionary and frequency filter are learned from training
% rows only. This avoids imposing numerical distances between arbitrary IDs.

arguments
    topK (1,1) struct
    trainIdx
    rows
    opts.MinTrainingObservations (1,1) double {mustBeInteger,mustBePositive} = 30
    opts.MaxLTEIdentities (1,1) double {mustBeInteger,mustBeNonnegative} = 64
    opts.MaxWLANIdentities (1,1) double {mustBeInteger,mustBeNonnegative} = 256
    opts.LTEMissingFloor_dBm (1,1) double = -140
    opts.WLANMissingFloor_dBm (1,1) double = -105
    opts.IncludeDetectionMask (1,1) logical = true
end

n = size(topK.LTE_IDs,1);
trainIdx = normalizeRows(trainIdx,n);
rows = normalizeRows(rows,n);
[lteDict,lteCount] = trainingDictionary(topK.LTE_IDs(trainIdx,:), ...
    opts.MinTrainingObservations, opts.MaxLTEIdentities);
[wlanDict,wlanCount] = trainingDictionary(topK.WLAN_IDs(trainIdx,:), ...
    opts.MinTrainingObservations, opts.MaxWLANIdentities);

XL = encodeTechnology(topK.LTE_IDs(rows,:), topK.LTE_RSRP_dBm(rows,:), ...
    lteDict, opts.LTEMissingFloor_dBm, opts.IncludeDetectionMask);
XW = encodeTechnology(topK.WLAN_IDs(rows,:), topK.WLAN_RSSI_dBm(rows,:), ...
    wlanDict, opts.WLANMissingFloor_dBm, opts.IncludeDetectionMask);

% Rank powers preserve strongest-signal ordering even for identities outside
% the retained dictionary.
rankPower = [fillmissing(single(topK.LTE_RSRP_dBm(rows,:)), 'constant', ...
    single(opts.LTEMissingFloor_dBm)), ...
    fillmissing(single(topK.WLAN_RSSI_dBm(rows,:)), 'constant', ...
    single(opts.WLANMissingFloor_dBm))];
X = [XL, XW, rankPower];

info = struct('LTE_ID',lteDict,'LTE_TrainingOccurrences',lteCount, ...
    'WLAN_ID',wlanDict,'WLAN_TrainingOccurrences',wlanCount, ...
    'IncludesDetectionMask',opts.IncludeDetectionMask);
end

function [dict, counts] = trainingDictionary(ids, minCount, maxCount)
v = double(ids(:)); v = v(v~=0);
if isempty(v) || maxCount==0, dict=zeros(0,1,'uint32'); counts=[]; return; end
[u,~,g] = unique(v); c = accumarray(g,1);
keep = c >= minCount; u=u(keep); c=c(keep);
[c,ord] = sort(c,'descend'); u=u(ord);
k=min(maxCount,numel(u)); dict=uint32(u(1:k)); counts=c(1:k);
end

function X = encodeTechnology(ids,power,dict,floorValue,withMask)
n=size(ids,1); d=numel(dict);
Xpower=repmat(single(floorValue),n,d); detected=false(n,d);
for k=1:size(ids,2)
    [tf,col]=ismember(ids(:,k),dict);
    row=find(tf & isfinite(power(:,k)));
    if ~isempty(row)
        linear=sub2ind([n,d],row,col(row));
        Xpower(linear)=single(power(row,k)); detected(linear)=true;
    end
end
if withMask, X=[Xpower,single(detected)]; else, X=Xpower; end
end

function idx=normalizeRows(idx,n)
if islogical(idx), assert(numel(idx)==n); idx=find(idx); else, idx=idx(:); end
end
