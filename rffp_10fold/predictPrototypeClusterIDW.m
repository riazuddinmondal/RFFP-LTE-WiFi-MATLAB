function [YPred,model] = predictPrototypeClusterIDW(XTrain,YTrain,XTest,method,cfg)
%PREDICTPROTOTYPECLUSTERIDW Scalable inductive AHC/FCM + prototype IDW.
% Test samples are never included in clustering.

rng(cfg.RandomSeed,'twister');
m=min(cfg.NumPrototypes,size(XTrain,1));
[protoLabel,C]=kmeans(double(XTrain),m,'MaxIter',cfg.KMeansMaxIter, ...
    'Replicates',1,'Display','off','EmptyAction','singleton');
CY=zeros(m,2); for j=1:m, CY(j,:)=mean(YTrain(protoLabel==j,:),1); end
k=min(cfg.NumClusters,max(2,m-1));

switch upper(string(method))
    case "AHC"
        Z=linkage(C,'ward','euclidean'); clusterLabel=cluster(Z,'maxclust',k);
        clusterCenters=zeros(k,size(C,2));
        for j=1:k, clusterCenters(j,:)=mean(C(clusterLabel==j,:),1); end
    case "FCM"
        [clusterCenters,U]=fcm(C,k,[cfg.FCMExponent cfg.FCMMaxIter cfg.FCMTolerance false]);
        [~,clusterLabel]=max(U,[],1); clusterLabel=clusterLabel(:);
    otherwise
        error('Method must be AHC or FCM.');
end

testCluster=knnsearch(clusterCenters,double(XTest),'K',1);
YPred=zeros(size(XTest,1),2);
for j=1:k
    q=find(testCluster==j); candidates=find(clusterLabel==j);
    if isempty(q),continue;end
    if isempty(candidates),candidates=(1:m)';end
    localK=min(cfg.KNNK,numel(candidates));
    [nn,d]=knnsearch(C(candidates,:),double(XTest(q,:)),'K',localK);
    nn=candidates(nn); w=1./max(d,1e-6); w=w./sum(w,2);
    for z=1:numel(q),YPred(q(z),:)=sum(CY(nn(z,:),:).*w(z,:)',1);end
end
model=struct('Prototypes',C,'PrototypeXY',CY,'ClusterLabel',clusterLabel, ...
    'ClusterCenters',clusterCenters,'Method',char(method));
end
