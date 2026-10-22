function T = reproduce_saved_metrics(resultsDir, outputDir)
% Recompute the primary error table from saved out-of-fold outputs, no fitting.
arguments
    resultsDir (1,1) string
    outputDir (1,1) string = "reproduced_results"
end
if ~isfolder(outputDir),mkdir(outputDir);end
B=load(fullfile(resultsDir,'corrected_route10fold_checkpoint.mat'),'results','savedFoldIndex');
N=numel(B.savedFoldIndex);assert(N==118930);fold=double(B.savedFoldIndex(:));
methods=["LSBoost","KNN","AHC","FCM","TunedDNN","HierarchicalFastKAN"];
errors=nan(N,numel(methods));
for m=1:4
    seen=false(N,1);
    for j=1:numel(B.results)
        r=B.results(j);
        if string(r.Representation)=="C" && string(r.Method)==methods(m)
            ix=find(fold==r.Fold);a=r.Metrics.Error_m(:);
            assert(numel(ix)==numel(a) && ~any(seen(ix)));
            errors(ix,m)=a;seen(ix)=true;
        end
    end
    assert(all(seen));
end
seen=false(N,1);trueXY=nan(N,2);
for k=1:10
    P=load(fullfile(resultsDir,'dnn_predictions',sprintf('C_TunedDNN_F%02d_predictions.mat',k)));
    ix=double(P.testRows(:));assert(~any(seen(ix)) && all(fold(ix)==k));
    errors(ix,5)=sqrt(sum((double(P.pred)-double(P.YTest)).^2,2));
    trueXY(ix,:)=P.YTest;seen(ix)=true;
end
assert(all(seen));
K=load(fullfile(resultsDir,'fastkan','FastKAN_V2_10fold_combined_results.mat'),'combined');
ix=double(K.combined.TestIndex(:));assert(numel(unique(ix))==N && all(fold(ix)==K.combined.Fold(:)));
assert(max(abs(trueXY(ix,:)-double(K.combined.TrueXY_m)),[],'all')<0.5);
errors(ix,6)=sqrt(sum((double(K.combined.PredictedXY_m)-double(K.combined.TrueXY_m)).^2,2));
assert(all(isfinite(errors),'all'));
Mean_m=mean(errors,1).';Median_m=median(errors,1).';RMSE_m=sqrt(mean(errors.^2,1)).';
P68_m=zeros(6,1);P95_m=zeros(6,1);Maximum_m=max(errors,[],1).';
for m=1:6,P68_m(m)=linearQ(errors(:,m),.68);P95_m(m)=linearQ(errors(:,m),.95);end
Method=methods.';NumSamples=repmat(N,6,1);
T=table(Method,NumSamples,Mean_m,Median_m,RMSE_m,P68_m,P95_m,Maximum_m);
writetable(T,fullfile(outputDir,'pooled_metrics.csv'));
save(fullfile(outputDir,'pooled_errors.mat'),'errors','methods','fold','-v7.3');disp(T);
end
function q=linearQ(e,p)
e=sort(double(e(:)));r=1+(numel(e)-1)*p;lo=floor(r);hi=ceil(r);
q=e(lo)+(r-lo)*(e(hi)-e(lo));
end
