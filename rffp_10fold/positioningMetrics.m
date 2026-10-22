function M = positioningMetrics(YTrue,YPred)
%POSITIONINGMETRICS Standard 2-D Euclidean positioning statistics.
e=sqrt(sum((double(YTrue)-double(YPred)).^2,2));
M=struct('Error_m',e,'Mean_m',mean(e,'omitnan'),'Median_m',median(e,'omitnan'), ...
    'Std_m',std(e,'omitnan'),'RMSE_m',sqrt(mean(e.^2,'omitnan')), ...
    'P68_m',prctile(e,68),'P95_m',prctile(e,95),'Maximum_m',max(e));
end
