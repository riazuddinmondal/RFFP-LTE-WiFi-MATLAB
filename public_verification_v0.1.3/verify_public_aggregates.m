function verify_public_aggregates(folder)
% MATLAB R2024a. Checks independently recomputable aggregate identities.
if nargin<1 || isempty(folder), folder=fileparts(mfilename('fullpath')); end
f=readtable(fullfile(folder,'tenfold_fold_sufficient_statistics.csv'),'TextType','string');
p=readtable(fullfile(folder,'tenfold_pooled_metrics.csv'),'TextType','string');
methods=unique(f.Method,'stable'); assert(numel(methods)==8);
for i=1:numel(methods)
 m=methods(i); a=f(f.Method==m,:); b=p(p.Method==m,:);
 assert(height(a)==10 && height(b)==1);
 N=sum(a.N); assert(N==118930);
 avg=sum(a.SumError_m)/N; rmse=sqrt(sum(a.SumSquaredError_m2)/N);
 assert(abs(avg-b.Mean_m)<1e-7);assert(abs(rmse-b.RMSE_m)<1e-7);
 fprintf('%s: N=%d mean=%.6f RMSE=%.6f PASS\n',m,N,avg,rmse);
end
c=readtable(fullfile(folder,'tenfold_threshold_counts_by_fold.csv'),'TextType','string');
for i=1:height(c)
 assert(c.Within10m(i)<=c.Within25m(i) && c.Within25m(i)<=c.Within50m(i));
 assert(c.Within50m(i)<=c.Within100m(i) && c.Within100m(i)<=c.N(i));
end
q=readtable(fullfile(folder,'fold_campaign_device_composition.csv'),'TextType','string');
assert(height(q)==11);assert(sum(q.TotalObservations(1:10))==118930);
fprintf('Aggregate arithmetic checks passed.\n');
fprintf('No independent test of original predictions, log assignments, or exact percentiles is implied.\n');
end
