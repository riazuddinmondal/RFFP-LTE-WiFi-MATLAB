%% Compare independent tuned DNN Fold 1 with archived predictions
clear; clc;
root='C:\RFFP_Release_Validation';
oldFile=fullfile(root,'private_data','results','dnn_predictions','C_TunedDNN_F01_predictions.mat');
newFile=fullfile(root,'test_outputs','DNN_Level3_Fold1','C_TunedDNN_F01_predictions.mat');
assert(isfile(oldFile),'Archived DNN prediction file missing: %s',oldFile);
assert(isfile(newFile),'Independent DNN prediction file missing: %s',newFile);
A=load(oldFile);B=load(newFile);
assert(isfield(A,'pred') && isfield(B,'pred'),'Prediction variable pred missing');
assert(isfield(A,'YTest') && isfield(B,'YTest'),'YTest missing');
assert(isequal(size(A.pred),size(B.pred)),'Prediction dimensions differ');
assert(isequal(A.YTest,B.YTest),'Test targets/order differ');
if isfield(A,'testRows') && isfield(B,'testRows')
    assert(isequal(A.testRows,B.testRows),'Test-row order differs');
end
if isfield(A,'testLogID') && isfield(B,'testLogID')
    assert(isequal(A.testLogID,B.testLogID),'Test-log order differs');
end
assert(all(isfinite(A.pred(:))) && all(isfinite(B.pred(:))),'Nonfinite predictions');
eA=sqrt(sum((double(A.pred)-double(A.YTest)).^2,2));
eB=sqrt(sum((double(B.pred)-double(B.YTest)).^2,2));
metricNames={'Mean_m','Median_m','RMSE_m','P68_m','P95_m','Maximum_m'};
calc=@(e) [mean(e),median(e),sqrt(mean(e.^2)),prctile(e,68),prctile(e,95),max(e)];
% Both sets use the same calculation; see original positioningMetrics for official percentile convention.
a=calc(eA);b=calc(eB);
T=table(string(metricNames(:)),a(:),b(:),abs(a(:)-b(:)), ...
    'VariableNames',{'Metric','Original_m','Reproduced_m','Difference_m'});
disp(T);
fprintf('Max individual error difference: %.9f m\n',max(abs(eA-eB)));
fprintf('Mean absolute difference in errors: %.9f m\n',mean(abs(eA-eB)));
fprintf('Sample count: %d\n',numel(eB));
% GPU stochasticity means nonzero differences are not an automatic failure.
% Review differences and original candidate/epoch selection before declaring pass.
report=struct('Method','TunedDNN','Fold',1,'Comparison',T, ...
    'MaxErrorDifference_m',max(abs(eA-eB)), ...
    'MeanAbsoluteErrorDifference_m',mean(abs(eA-eB)), ...
    'NumTest',numel(eB),'RequiresScientificReview',true);
save(fullfile(root,'test_outputs','DNN_Level3_Comparison_Report.mat'),'report');
