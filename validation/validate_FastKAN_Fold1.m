%% Compare independent FastKAN V2 Fold 1 with archived results
clear; clc;
root='C:\RFFP_Release_Validation';
oldFile=fullfile(root,'private_data','results','fastkan','FastKAN_V2_fold1_results.mat');
newFile=fullfile(root,'test_outputs','FastKAN_Level3_Fold1','FastKAN_V2_fold1_results.mat');
assert(isfile(oldFile),'Archived FastKAN result missing: %s',oldFile);
assert(isfile(newFile),'Independent FastKAN result missing: %s',newFile);
A=load(oldFile,'results');B=load(newFile,'results');
a=A.results;b=B.results;
assert(a.Fold==1 && b.Fold==1,'Incorrect fold');
assert(isequal(a.TestIndex(:),b.TestIndex(:)),'Test indices/order differ');
assert(numel(a.PositionError_m)==numel(b.PositionError_m),'Error count mismatch');
eA=double(a.PositionError_m(:));eB=double(b.PositionError_m(:));
assert(all(isfinite(eA)) && all(isfinite(eB)),'Nonfinite positioning errors');
calc=@(e) [mean(e),median(e),sqrt(mean(e.^2)),prctile(e,68),prctile(e,95),max(e)];
metricNames={'Mean_m','Median_m','RMSE_m','P68_m','P95_m','Maximum_m'};
va=calc(eA);vb=calc(eB);
T=table(string(metricNames(:)),va(:),vb(:),abs(va(:)-vb(:)), ...
    'VariableNames',{'Metric','Original_m','Reproduced_m','Difference_m'});
disp(T);
fprintf('Maximum individual error difference: %.9f m\n',max(abs(eA-eB)));
fprintf('Mean absolute error difference: %.9f m\n',mean(abs(eA-eB)));
fprintf('Samples compared: %d\n',numel(eB));
if isfield(a,'Config') && isfield(b,'Config')
    fprintf('Original seed: %g | New seed: %g\n',a.Config.Seed,b.Config.Seed);
end
% No automatic pass: inspect stochastic GPU differences and settings.
report=struct('Method','HierarchicalFastKAN','Fold',1,'Comparison',T, ...
    'MaxErrorDifference_m',max(abs(eA-eB)), ...
    'MeanAbsoluteErrorDifference_m',mean(abs(eA-eB)), ...
    'NumTest',numel(eB),'RequiresScientificReview',true);
save(fullfile(root,'test_outputs','FastKAN_Level3_Comparison_Report.mat'),'report');
