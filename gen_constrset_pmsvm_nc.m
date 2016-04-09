function MCs = gen_constrset_pmsvm_nc(X, numPairs,incrFeatures,decFeatures,params)
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
%               - nCandidates: size of random subset used in creation of
%               each contraint.
%                   '5' was used for the ECMLPKDD paper.
%               - nAdditionalMidPts: number of mid points to add between base point
%                   feature value and subset min/max. Usually set to 0.
%               - GreedyMaxMinSeparation:'0' to ignore (just use random
%                   subsequent constraints), or '1' to use greedy maximum
%                   minimum separation to selection each subsequent constraint
%                   (does its best to spread the constraints through the space)
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
    Xcopy=X;
    n=size(Xcopy,1);
    nCandidates=str2num(params{1});
    nAdditionalMidPts=str2num(params{2});
    GreedyMaxMinSeparation=str2num(params{3});
    boolDuplicateDetection=str2num(params{4});
    nCandidates=min(size(X,1),nCandidates); % fix to number of data points available
    constrFeats=[incrFeatures decFeatures];
    myrandstrm= RandStream('mcg16807','seed','shuffle');
%     % reverse decreasing features and correct at the end of the function
%     for i=1:numel(decFeatures)
%        Xcopy(:,decFeatures(i))= -Xcopy(:,decFeatures(i));
%     end
 % 15: CALCULATE STANDARDISATION PARAMS
    stdevs=zeros(size(Xcopy,2),1);
    means=zeros(size(Xcopy,2),1);
    for i=1:size(Xcopy,2)
        stdevs(i,1)=iif(std(Xcopy(:,i))==0,1,std(Xcopy(:,i)));
        means(i,1)=mean(Xcopy(:,i));
    end
    Xcopy_std=(Xcopy-repmat(means(:,1)',[size(Xcopy,1) 1]))./repmat(stdevs(:,1)',[size(Xcopy,1) 1]);
    % create constraints  
    MCs=zeros(numPairs,size(Xcopy,2),2);
    iNextConstraintID=1;
    numduplicatesInARow=0;
    basepts_inds_used_all=containers.Map(); %zeros(0,numel(constrFeats));
    for ifeat=1:numel(constrFeats)
        basepts_inds_used_all(num2str(ifeat))=[];
    end
    while iNextConstraintID<=numPairs && numduplicatesInARow<500
        shuf=randperm(numel(constrFeats));
        for ifeat=shuf
            feat=constrFeats(ifeat);
            [nfeatparts,featparts]=get_partitions(Xcopy(:,feat),100,true,true);
            if nfeatparts>0 % make sure there is at least some variation in this feature 
                nfeatparts=iif(nfeatparts>=10,0,nfeatparts); % assign nfeatparts=0 to continuous variables, otherwise, featparts contains viable feature values
                %select base point at random
                if GreedyMaxMinSeparation==0 % random
                    basepts_inds_used_feat=basepts_inds_used_all(num2str(ifeat));
                    basepts_inds_avail=setdiff(1:size(Xcopy,1),basepts_inds_used_feat);
                    if numel(basepts_inds_avail)==0 && nCandidates>=size(Xcopy,1)
                        baseid=0;
                    elseif numel(basepts_inds_avail)==0 % just randomly select points now
                        baseid=randi(size(Xcopy,1));
                        Xbase=Xcopy(baseid,:);
                    else
                        baseid=basepts_inds_avail(randi(numel(basepts_inds_avail)));
                        Xbase=Xcopy(baseid,:);
                        basepts_inds_used_all(num2str(ifeat))=[basepts_inds_used_all(num2str(ifeat)) baseid];
                    end
                    %basepts_inds_used(end+1,ifeat)=baseid;
                else % use greedy max-min-sep
                    basepts_inds_used_feat=basepts_inds_used_all(num2str(ifeat));
                    basepts_inds_avail=setdiff(1:size(Xcopy,1),basepts_inds_used_feat);
                    avail_pts=Xcopy(basepts_inds_avail,:);
                    avail_pts_std=Xcopy_std(basepts_inds_avail,:); %(avail_pts-repmat(means(:,1)',[size(avail_pts,1) 1]))./repmat(stdevs(:,1)',[size(avail_pts,1) 1]);
                    if numel(basepts_inds_used_feat)==0  % no previous points chosen. Use a random one to start.
                        newpt=randi(size(avail_pts,1));
                        basepts_inds_used_all(num2str(ifeat))=[newpt];
                    else % find max-min-sep pt
                        maxmindist=-9e9;
                        selpt=-9e9;
                        if numel(basepts_inds_avail)==0 && nCandidates>=size(Xcopy,1) % no point addding more constraints, already added all possible
                            newpt=0;
                        elseif  numel(basepts_inds_avail)==0 && nCandidates<size(Xcopy,1) % already been all the way round, just select a random point
                            newpt=randi(size(Xcopy,1));
                        else
                            for p=1:numel(basepts_inds_avail)
                               diffs= Xcopy_std(basepts_inds_used_feat,:)-repmat(avail_pts_std(p,:),[numel(basepts_inds_used_feat) 1]);
                               dists=sum(diffs.*diffs,2);
                               mindist=min(dists);
                               if mindist>maxmindist
                                   maxmindist=mindist;
                                   selpt=basepts_inds_avail(p);
                               end
                            end
                            newpt=selpt;
                            basepts_inds_used_all(num2str(ifeat))=[basepts_inds_used_all(num2str(ifeat)) newpt];
                        end

                    end
                    baseid=newpt;
                    if baseid>0
                        Xbase=Xcopy(baseid,:);
                    end
                end
                if baseid==0 % no pmore constraints to add for this feature
                    iNextConstraintID=iNextConstraintID+1;
                else
                    %select candidate co-hort
                    foundDiffering=false;
                    while ~foundDiffering
                        Xcandidate=datasample(myrandstrm,Xcopy,nCandidates-1,'Replace',false);
                        if min(Xcandidate(:,feat))~=max(Xcandidate(:,feat))
                            foundDiffering=true;
                        end 
                    end
                    % select one data point at random from the candidates to be the
                    % base data point for this constraint
                    %baseid=randi(nCandidates);
                    %Xbase=Xcandidate(baseid,:);
                    Xcandidate(end+1,:)=Xbase; % add base point
                    minFeatVal=min(Xcandidate(:,feat));
                    maxFeatVal=max(Xcandidate(:,feat));
                    featVal=Xbase(1,feat);
                    boolIncreasing=sum(decFeatures==feat)<=0;
                    for dirn=[-1,1] % construct constraints with requested mid ptss in each direction
                        constrExtents=iif(dirn==1,[featVal maxFeatVal],[minFeatVal featVal]);
                        constrExtents=sort(unique(constrExtents),'ascend');
                        if numel(constrExtents)==1
                            % we are already at an extreme, cannot create a
                            % constraint so do nothing.
                        else % create our constraints
                            if nAdditionalMidPts>0 % have additional mid points to add
                                if nfeatparts==0 % we have a continuous feature
                                    constrExtents=constrExtents(1):(constrExtents(2)-constrExtents(1))/(nAdditionalMidPts+1) :constrExtents(2);
                                else % we have a categorical variable
                                    availablevals=iif(dirn==1,featparts(featparts>=featVal),featparts(featparts<=featVal));
                                    availablevals=sort(unique([availablevals featVal]),'ascend');
                                    [npartconst, constrExtents]=get_partitions(availablevals,nAdditionalMidPts+1,true,true);
                                    %constrExtents=constrExtents';
                                end
                            end
                            % create constraints between numbers in constrExtents
                             % create constraints between midspread pts
                            for md=1:numel(constrExtents)-1
                                % construct constraint points
                                constr1=Xbase;
                                constr2=Xbase;
                                if boolIncreasing
                                     constr1(1,feat)= constrExtents(1,md);
                                     constr2(1,feat)= constrExtents(1,md+1); 
                                else % decreasing
                                     constr1(1,feat)= constrExtents(1,md+1);
                                     constr2(1,feat)= constrExtents(1,md); 
                                end
                                % check whether this is a virtual duplicate of an
                                % existing constraint
                                if boolDuplicateDetection==0 || iNextConstraintID==1 % haven't got any yet
                                    isduplicate=0;
                                else % have some existing constraints
                                    deviations_abs=abs([MCs(1:iNextConstraintID-1,:,1) MCs(1:iNextConstraintID-1,:,2)]- repmat([constr1 constr2],iNextConstraintID-1,1));
                %                     if strcmp(feattype,'cat')==1 % categorical
                                    toocloseSDs_factor=0.2;
                                    tooclose=deviations_abs<repmat(toocloseSDs_factor*[stdevs' stdevs'],size(deviations_abs,1),1);
                                    [isduplicate,indx]=ismember(logical(ones(1,size(deviations_abs,2))),tooclose,'rows'); % ie if there are any existing rows with all values <0.1 stdevs, we have a duplicate
                                    if isduplicate
                                        numduplicatesInARow=numduplicatesInARow+1;
                                        %disp('duplicate found');
                                    end
                                end
                                % add constraint
                                if ~isduplicate
                                     numduplicatesInARow=0; % reset counter
                                    MCs(iNextConstraintID,:,1)=constr1;
                                     MCs(iNextConstraintID,:,2)=constr2;
                                     iNextConstraintID=iNextConstraintID+1;
                                end
                           end
                        end
                    end
                end
            end
        end
    end
    % trim additional points
    MCs=MCs(1:numPairs,:,:);
end
