%% RFFP Level 3: independent LSBoost Fold 1 checkpoint comparison
clear; clc;
root = 'C:\RFFP_Release_Validation';
originalFile = fullfile(root,'reference','corrected_route10fold_checkpoint.mat');
newFile = fullfile(root,'test_outputs','RFFP_Level3_LSBoost_Fold1.mat');
assert(isfile(originalFile),'Missing original checkpoint: %s',originalFile);
assert(isfile(newFile),'Missing new checkpoint: %s',newFile);
A = load(originalFile,'results','savedFoldIndex','savedConfig');
B = load(newFile,'results','savedFoldIndex','savedConfig');
assert(isequal(A.savedFoldIndex,B.savedFoldIndex), 'Fold assignments differ');
assert(isequaln(A.savedConfig,B.savedConfig), 'Scientific configuration differs');
key = "C_LSBoost_F01";
ia = find(string({A.results.Key}) == key);
ib = find(string({B.results.Key}) == key);
assert(numel(ia)==1 && numel(ib)==1,'Expected exactly one %s record in each checkpoint',key);
a = A.results(ia); b = B.results(ib);
assert(a.Fold==1 && b.Fold==1,'Incorrect fold');
assert(strcmp(string(a.Representation),"C") && strcmp(string(b.Representation),"C"),'Incorrect representation');
assert(strcmp(string(a.Method),"LSBoost") && strcmp(string(b.Method),"LSBoost"),'Incorrect method');
assert(a.NumTrain==b.NumTrain && a.NumTest==b.NumTest,'Sample counts differ');
metrics = {'Mean_m','Median_m','RMSE_m','P68_m','P95_m','Maximum_m'};
tolerance = 1e-6;
orig = zeros(numel(metrics),1); fresh = orig;
for k=1:numel(metrics)
    orig(k)=a.Metrics.(metrics{k});
    fresh(k)=b.Metrics.(metrics{k});
end
assert(all(isfinite(orig)) && all(isfinite(fresh)),'Nonfinite summary metric');
difference = abs(orig-fresh);
T=table(string(metrics(:)),orig,fresh,difference, ...
    'VariableNames',{'Metric','Original_m','Reproduced_m','AbsoluteDifference_m'});
fprintf('\nIndependent LSBoost C Fold 1 validation\n');
fprintf('Train: %d; Test: %d\n',b.NumTrain,b.NumTest);
disp(T);
fprintf('Maximum metric difference: %.12g m\n',max(difference));
vectorCompared=false; vectorDifference=NaN;
if isfield(a.Metrics,'Error_m') && isfield(b.Metrics,'Error_m')
    ea=a.Metrics.Error_m(:); eb=b.Metrics.Error_m(:);
    assert(numel(ea)==numel(eb) && numel(ea)==b.NumTest,'Error-vector length mismatch');
    assert(all(isfinite(ea)) && all(isfinite(eb)),'Nonfinite error vector');
    vectorDifference=max(abs(ea-eb));
    vectorCompared=true;
    fprintf('Maximum individual error difference: %.12g m\n',vectorDifference);
else
    fprintf('Individual error vectors not available for both records.\n');
end
fprintf('New execution runtime: %.2f seconds (%.2f minutes)\n',b.Runtime_s,b.Runtime_s/60);
metricsPass=all(difference<tolerance);
vectorPass=~vectorCompared || vectorDifference<tolerance;
passed=metricsPass && vectorPass;
report=struct('Key',key,'NumTrain',b.NumTrain,'NumTest',b.NumTest, ...
    'Tolerance_m',tolerance,'ComparisonTable',T, ...
    'MaximumMetricDifference_m',max(difference), ...
    'ErrorVectorCompared',vectorCompared, ...
    'MaximumErrorVectorDifference_m',vectorDifference, ...
    'Runtime_s',b.Runtime_s,'ValidationPassed',passed, ...
    'ValidationDate',datetime('now'));
reportFile=fullfile(root,'test_outputs','RFFP_Level3_LSBoost_Validation_Report.mat');
save(reportFile,'report');
fprintf('Report saved: %s\n',reportFile);
if passed
    fprintf('PASS: LSBoost Fold 1 numerical reproducibility verified.\n');
else
    fprintf('CHECK REQUIRED: LSBoost ran, but numerical reproducibility tolerance was exceeded.\n');
    fprintf('Do not modify scientific settings to force agreement; investigate first.\n');
end
