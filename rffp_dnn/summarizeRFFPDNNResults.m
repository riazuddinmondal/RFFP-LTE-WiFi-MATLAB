function summary = summarizeRFFPDNNResults(results)
%SUMMARIZERFFPDNNRESULTS One row per completed DNN outer fold.
if isempty(results)
    summary = table();
    return;
end
n = numel(results);
Representation = strings(n,1); Fold=zeros(n,1); NumTrain=zeros(n,1);
NumTest=zeros(n,1); BestEpoch=zeros(n,1); Runtime_s=zeros(n,1);
MeanPE_m=zeros(n,1); MedianPE_m=zeros(n,1); RMSE_m=zeros(n,1);
P68_m=zeros(n,1); P95_m=zeros(n,1); MaximumPE_m=zeros(n,1);
for i=1:n
    Representation(i)=string(results(i).Representation);
    Fold(i)=results(i).Fold; NumTrain(i)=results(i).NumTrain;
    NumTest(i)=results(i).NumTest; BestEpoch(i)=results(i).BestEpoch;
    Runtime_s(i)=results(i).Runtime_s; M=results(i).Metrics;
    MeanPE_m(i)=M.Mean_m; MedianPE_m(i)=M.Median_m;
    RMSE_m(i)=M.RMSE_m; P68_m(i)=M.P68_m; P95_m(i)=M.P95_m;
    MaximumPE_m(i)=M.Maximum_m;
end
summary=table(Representation,Fold,NumTrain,NumTest,BestEpoch,Runtime_s, ...
    MeanPE_m,MedianPE_m,RMSE_m,P68_m,P95_m,MaximumPE_m);
summary=sortrows(summary,{'Representation','Fold'});
end

