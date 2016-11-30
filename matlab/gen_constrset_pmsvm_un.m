function MCs = gen_constrset_pmsvm_un(X, numBasePtsPerFeat,incrFeatures,decFeatures,params)
%gen_constrset_pmsvm_nc - Generates a set of non-conjunctive (univariate)
%constraints for use with constrained SVM.
% DESCRIPTION:
%   This generates a MxPx2 matrix which represents a set of M constraints to be 
%   used with the constraned SVM algorithm (train_consvm_rbf). Each constraint is
%   the pair of points (x_m,x_m') where x_m=MCs(m,:,1) and x_m'=MCs(m,:,2).
%   train_consvm_rbf() can then be used to train an SVM for which f(x_m')>=f(x_m) 
%   is guaranteed for this set of constraints.
%   This algorithm is used for NCr and NCm contraints as described in Bartley 
%   et al. 2016 'Effective Knowledge Integration in Support Vector Machines 
%   for Improved Accuracy' (submitted to ECMLPKDD2016).
%
%   Creates constraints based on random points from the training data.
%   Each random 'base' point is used to create at two constraints,
%   each between the base point and the maximum value seen in nCandidate
%   points, and between the base point and the minimum value seen.
%   The second parameter allows the addition of nAdditionalMidPts points
%   which are evenly spread along the constraint extent.
%
% INPUTS:
%    X - NxP matrix of training data
%    numPairs - the number of constraints to be created (M)
%    incrFeatures - p_inc scalar indicating which features (p_inc subset of
%    (1..P)) are INCREASING monotone
%    decFeatures - p_dec scalar indicating which features (p_dec subset of
%    (1..P)) are DECREASING monotone
%    params - set of options {nCandidates,nAdditionalMidPts,GreedyMaxMinSeparation
%           boolDuplicateDetection} where:
%               - nAdditionalMidPts: number of mid points to add between base point
%                   feature value and subset min/max. Usually set to 0.
%               - boolDuplicateDetection: '0' for none, or '1' for
%                   duplicate detection. '1' is recommended and was used for the 
%                   ECMLPKDD paper
%               
% OUTPUTS:
%    MCs - MxPx2 matrix representing a set of M constraints to be 
%   used with the constraned SVM algorithm (train_consvm_rbf). Each constraint is
%   the pair of points (x_m,x_m') where x_m=MCs(m,:,1) and x_m'=MCs(m,:,2).
%
% EXAMPLE:
%   To create 500 constraints, with increasing features [2 5] and
%   decreasing features [3 7], with no mid pts but with greedy mms and duplicate 
%   detection, and then solve the PM-SVM model:
%   MCs=gen_constrset_pmsvm_nc(Xtrain, 500,[2 5],[3 7],{'5','0','1','1'}) 
%   [alphas,betas, y_pred,bias,H, minEig] = train_consvm_rbf(ytrain,Xtrain,bc,kf,Xtest, MCs)
% Other m-files required: get_partitions, iif
%
% See also: train_consvm_rbf

% Author: Chris Bartley
% University of Western Australia, School of Computer Science
% email address: christopher.bartley@research.uwa.edu.au
% Website: http://staffhome.ecm.uwa.edu.au/~19514733/
% Last revision: 30-March-2016

%------------- BEGIN CODE --------------
    Xcopy=unique(X,'rows');
    n=size(Xcopy,1);
    nAdditionalMidPts=str2num(params{1});
    boolDuplicateDetection=str2num(params{2});
    constrFeats=[incrFeatures decFeatures];
    myrandstrm= RandStream('mcg16807','seed','shuffle');
 % 15: CALCULATE STANDARDISATION PARAMS
    stdevs=zeros(size(Xcopy,2),1);
    means=zeros(size(Xcopy,2),1);
    maxes=zeros(size(Xcopy,2),1);
    mins=zeros(size(Xcopy,2),1);
    for i=1:size(Xcopy,2)
        stdevs(i,1)=iif(std(Xcopy(:,i))==0,1,std(Xcopy(:,i)));
        means(i,1)=mean(Xcopy(:,i));
        maxes(i,1)=max(Xcopy(:,i));
        mins(i,1)=min(Xcopy(:,i));
    end
    Xcopy_std=(Xcopy-repmat(means(:,1)',[size(Xcopy,1) 1]))./repmat(stdevs(:,1)',[size(Xcopy,1) 1]);
    % create constraints  
    MCs=zeros(numBasePtsPerFeat*2*numBasePtsPerFeat,size(Xcopy,2),2);
    iNextConstraintID=1;
    same_val_indxs=[];
    for ifeat=1:numel(constrFeats)
        feat=constrFeats(ifeat);
        increasing=sum(decFeatures==feat)<=0;
        feat_max=maxes(feat,1);
        feat_min=mins(feat,1);
        % pick numBasePtsPerFeat base points
        basept_indxs=randi(n,[numBasePtsPerFeat,1]);
        mcs1=Xcopy(basept_indxs,:);
        
        MCs(iNextConstraintID:iNextConstraintID+numBasePtsPerFeat-1,:,1)=mcs1;
        MCs(iNextConstraintID:iNextConstraintID+numBasePtsPerFeat-1,:,2)=mcs1;
        MCs(iNextConstraintID:iNextConstraintID+numBasePtsPerFeat-1,feat,iif(increasing,1,2))=ones(numBasePtsPerFeat,1)*feat_min;
        MCs(iNextConstraintID+numBasePtsPerFeat:iNextConstraintID+2*numBasePtsPerFeat-1,:,1)=mcs1;
        MCs(iNextConstraintID+numBasePtsPerFeat:iNextConstraintID+2*numBasePtsPerFeat-1,:,2)=mcs1;
        MCs(iNextConstraintID+numBasePtsPerFeat:iNextConstraintID+2*numBasePtsPerFeat-1,feat,iif(increasing,2,1))=ones(numBasePtsPerFeat,1)*feat_max;
        % append indexes of constraints with same feature values
        same_val_indxs=[same_val_indxs; (iNextConstraintID-1+find(MCs(iNextConstraintID:iNextConstraintID+2*numBasePtsPerFeat-1,feat,1)==MCs(iNextConstraintID:iNextConstraintID+2*numBasePtsPerFeat-1,feat,2)))];
        iNextConstraintID=iNextConstraintID+2*numBasePtsPerFeat;
    end
    % trim additional points 
    MCs=MCs(1:iNextConstraintID-1,:,:);
    % remove constraints with same feat vals
    MCs=MCs(setdiff(1:size(MCs,1),same_val_indxs),:,:);
    % remove duplicates
    MCsall=[MCs(:,:,1) MCs(:,:,2)];
    MCsall=unique(MCsall,'rows');
    MCs_new=zeros([size(MCsall,1),size(MCs,2),2]);
    MCs_new(:,:,1)=MCsall(:,1:size(MCs,2));
    MCs_new(:,:,2)=MCsall(:,1+size(MCs,2):size(MCsall,2));
    MCs=MCs_new;
end
