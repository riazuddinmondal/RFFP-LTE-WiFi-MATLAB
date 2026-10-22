function candidates = makeRFFPDNNCandidates(base)
%MAKERFFPDNNCANDIDATES Reproducible, bounded DNN tuning search space.
% Candidate 1 is the published baseline. The remaining candidates vary one
% or two related capacity/regularization/optimization choices.

names = ["Baseline_512";"Narrow_256";"Wide_1024";"Deep_512"; ...
    "LowerLR";"HigherLR";"StrongReg";"LightReg";"Batch256";"Batch1024"];

candidates = repmat(setCandidate(base,[512 256 128 64], ...
    [.20 .15 .10],1e-3,1e-4,512,names(1)),10,1);
candidates(1) = setCandidate(base,[512 256 128 64],[.20 .15 .10],1e-3,1e-4,512,names(1));
candidates(2) = setCandidate(base,[256 128 64],[.20 .15],1e-3,1e-4,512,names(2));
candidates(3) = setCandidate(base,[1024 512 256 128],[.25 .20 .10],1e-3,2e-4,512,names(3));
candidates(4) = setCandidate(base,[512 512 256 128 64],[.25 .20 .15 .10],1e-3,2e-4,512,names(4));
candidates(5) = setCandidate(base,[512 256 128 64],[.20 .15 .10],5e-4,1e-4,512,names(5));
candidates(6) = setCandidate(base,[512 256 128 64],[.25 .15 .10],2e-3,2e-4,512,names(6));
candidates(7) = setCandidate(base,[512 256 128 64],[.35 .25 .15],1e-3,5e-4,512,names(7));
candidates(8) = setCandidate(base,[512 256 128 64],[.10 .05 .05],1e-3,1e-5,512,names(8));
candidates(9) = setCandidate(base,[512 256 128 64],[.20 .15 .10],1e-3,1e-4,256,names(9));
candidates(10)= setCandidate(base,[512 256 128 64],[.20 .15 .10],1e-3,1e-4,1024,names(10));
end

function c=setCandidate(base,hidden,dropout,lr,l2,batch,name)
c=base; c.HiddenUnits=hidden; c.Dropout=dropout;
c.InitialLearnRate=lr; c.L2Regularization=l2;
c.MiniBatchSize=batch; c.CandidateName=name;
end
