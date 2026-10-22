function summary = summarizeRFFPResults(results)
%SUMMARIZERFFPRESULTS Fold-mean summary without requiring groupsummary.
if isempty(results), summary=table(); return; end
rep=string({results.Representation})'; method=string({results.Method})';
combos=unique([rep method],'rows','stable');
n=size(combos,1); NumFolds=zeros(n,1); Mean_m=zeros(n,1); RMSE_m=zeros(n,1);
P68_m=zeros(n,1); P95_m=zeros(n,1); Runtime_s=zeros(n,1);
for i=1:n
    idx=rep==combos(i,1) & method==combos(i,2); r=results(idx); NumFolds(i)=numel(r);
    Mean_m(i)=mean(arrayfun(@(x)x.Metrics.Mean_m,r));
    RMSE_m(i)=mean(arrayfun(@(x)x.Metrics.RMSE_m,r));
    P68_m(i)=mean(arrayfun(@(x)x.Metrics.P68_m,r));
    P95_m(i)=mean(arrayfun(@(x)x.Metrics.P95_m,r));
    Runtime_s(i)=mean([r.Runtime_s]);
end
Representation=combos(:,1); Method=combos(:,2);
summary=table(Representation,Method,NumFolds,Mean_m,RMSE_m,P68_m,P95_m,Runtime_s);
end
