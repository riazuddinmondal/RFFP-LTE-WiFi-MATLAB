function YPred = predictIDWKNNBlockwise(XTrain,YTrain,XTest,K,blockSize)
%PREDICTIDWKNNBLOCKWISE KNN with inverse-distance weighting in test blocks.
arguments
    XTrain {mustBeNumeric}
    YTrain {mustBeNumeric}
    XTest {mustBeNumeric}
    K (1,1) double {mustBeInteger,mustBePositive}=5
    blockSize (1,1) double {mustBeInteger,mustBePositive}=500
end
YPred=zeros(size(XTest,1),size(YTrain,2));
searcher=createns(double(XTrain),'NSMethod','exhaustive','Distance','euclidean');
for s=1:blockSize:size(XTest,1)
    idx=s:min(s+blockSize-1,size(XTest,1));
    [nn,d]=knnsearch(searcher,double(XTest(idx,:)),'K',K);
    w=1./max(d,1e-6); w=w./sum(w,2);
    for q=1:numel(idx)
        YPred(idx(q),:)=sum(YTrain(nn(q,:),:).*w(q,:)',1);
    end
end
end
