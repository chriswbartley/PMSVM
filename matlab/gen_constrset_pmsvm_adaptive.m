function MCs = gen_constrset_pmsvm_adaptive(X, numPairs,incrFeatures,decFeatures,params)
%gen_constrset_pmsvm_adaptive - Generates a set of non-conjunctive (univariate)
% constraints for use with constrained SVM, based on knowledge of the unconstrained SVM
% model.
% DESCRIPTION:
%   This generates a MxPx2 matrix which represents a set of M constraints to be 
%   used with the constrained SVM algorithm (train_consvm_rbf). Each constraint is
%   the pair of points (x_m,x_m') where x_m=MCs(m,:,1) and x_m'=MCs(m,:,2).
%   train_consvm_rbf() can then be used to train an SVM for which f(x_m')>=f(x_m) 
%   is guaranteed for this set of constraints.
%   This algorithm is used for the AD adaptivecontraints as described in Bartley 
%   et al. 2016 'Effective Knowledge Integration in Support Vector Machines 
%   for Improved Accuracy' (submitted to ECMLPKDD2016).
%
%   In brief, it only creates constraints at training datapoints known to
%   be non-monotone in the unconstrained SVM model. See paper for more
%   details.
%
% INPUTS:
%    X - NxP matrix of training data
%    numPairs - the number of constraints to be created (M)
%    incrFeatures - p_inc scalar indicating which features (p_inc subset of
%    (1..P)) are INCREASING monotone
%    decFeatures - p_dec scalar indicating which features (p_dec subset of
%    (1..P)) are DECREASING monotone
%    params - set of options {BasePtSeln,FeatBudget,EndPtSeln,
%       BasePtAnchor,EndPtAnchor,BasePtSelnPriorCorr,
%       MCC_outlierdetection,BasePtsUnbounded,DuplicatesCheck,SVM_kf,
%       SVM_y_train,SVM_X_train,SVM_alphas,SVM_bias} where:
            % - BasePtSeln: Base point selection: 'randseln' for random, or 
            %       'maxminsep' to use greedy max min separation
            % - FeatBudget: How the total number of constraints is divided 
            %       the monotone features: either 'equalbudget' (each feature gets
            %       M/pc constraints, where pc is the number of constrained features) or
            %       'NMTpt_prop' (recommended) where they are allocated
            %       proportionally to the number of nonmonotone points for
            %       each constrained feature.
            % - EndPtSeln: How the end point for each constraint (in the Passive NMR) 
            %       is selected: 'randseln' (recommended: random within Passive NMR) or 
            %       'closestRnminus1' (NOT IMPLEMENTED).
            % - BasePtAnchor: 'baseanchor_none' (recommended),'baseanchor_both','baseanchor_smart'
            % - EndPtAnchor: 'endanchor_none' (recommended),'endanchor_both','endanchor_smart'
            % - BasePtSelnPriorCorr: whether to prioritise constraints at non-monotone 
            %       support vectors: 'base_noprior' (no priority - recommended), or 
            %       'base_priorSVs' (prioritise non-monotone support
            %       vectors)
            % - MCC_outlierdetection: the non-monotonicity of each point is checked to 
            %       the global feature minimum and maximum. Sometimes this can be unrealistic
            %       due to the presence of outlier feature values. This option ignores outliers 
            %       ('od_on' - recommended) or 'od_off' (no outlier
            %       detection, check out to global extrema).
            % - BasePtsUnbounded: Whether to stop creating constraints once all
            %       non-monotone points have  constraint ('ub_on'), otherwise 
            %       it will continue 'ub_off' (recommended)
            % - DuplicatesCheck: whether to check and prevent duplicate constraints
            %       being created 'dupchk_off' or 'dupchk_on' (recommended)
            % - SVM_kf: RBF kernel factor to use 
            % - SVM_y_train: SVM y_train
            % - SVM_X_train: SVM X_train
            % - SVM_alphas: SVM alphas
            % - SVM_bias: SVM bias
            
%               
% OUTPUTS:
%    MCs - MxPx2 matrix representing a set of M constraints to be 
%   used with the constraned SVM algorithm (train_consvm_rbf). Each constraint is
%   the pair of points (x_m,x_m') where x_m=MCs(m,:,1) and x_m'=MCs(m,:,2).
%
% EXAMPLE:
%   To create 500 constraints for increasing features [2 5] and
%   decreasing features [3 7], with recommended option settings, and then solve the PM-SVM model:
%   First solve unconstrained SVM:
%       [SVM_alphas,betas, y_pred,SVM_bias,H, minEig] = train_consvm_rbf(ytrain,Xtrain,bc,kf,Xtest, [])
%   Then create constraints:
%       MCs=gen_constrset_pmsvm_adaptive(Xtrain, 500,[2 5],[3 7],{'randseln','NMTpt_prop','randseln','baseanchor_none','endanchor_none','base_noprior','od_off','ub_on','dupchk_on',kf,ytrain,Xtrain,SVM_alphas,SVM_bias}) 
%   Then solve constrained SVM with constraints:
%       [alphas,betas, y_pred,bias,H, minEig] = train_consvm_rbf(ytrain,Xtrain,bc,kf,Xtest, MCs)
%
% Other m-files required:  iif, calc_mcc_pmsvm_rbf
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

    BasePtSeln=params(1);
    FeatBudget=params(2);
    EndPtSeln=params(3);
    BasePtAnchor=params(4);
    EndPtAnchor=params(5);
    BasePtSelnPriorCorr=params(6);
    MCC_outlierdetection=params(7);
    BasePtsUnbounded=params(8);
    DuplicatesCheck=params(9);
    constrFeats=[incrFeatures decFeatures];
    SVM_kf=params{10};
    SVM_y_train=params{11};
    SVM_X_train=params{12};
    SVM_alphas=params{13};
    SVM_bias=params{14};
     
    myrandstrm= RandStream('mcg16807','seed','shuffle');

    % 15: CALCULATE STANDARDISED X
    stdevs=zeros(size(Xcopy,2),1);
    means=zeros(size(Xcopy,2),1);
    for i=1:size(Xcopy,2)
        stdevs(i,1)=iif(std(Xcopy(:,i))==0,1,std(Xcopy(:,i)));
        means(i,1)=mean(Xcopy(:,i));
    end
    Xcopy_std=(Xcopy-repmat(means(:,1)',[size(Xcopy,1) 1]))./repmat(stdevs(:,1)',[size(Xcopy,1) 1]);
    % 20: ESTABLISH BASE PT POOL
    X_base_pt_pool=containers.Map();
    X_base_pt_pool_std=containers.Map();
    y_base_pt_pool=containers.Map();
    alphas_base_pt_pool=containers.Map();
    ypred_base_pt_pool=containers.Map();
    X_NMT_Info=containers.Map(); % column 1 gives NMT direction: -1 means NMT prior, +1 means NMT after
    uniqfeatvals=containers.Map();
    feat_outlierlims=zeros(2,size(Xcopy,2));
    feattypes=containers.Map();
    for ff=1:numel(constrFeats)
        imonoFeat=constrFeats(ff);
        boolIncreasing=any(abs(incrFeatures-imonoFeat)<1e-10);
        vals=sort(X(:,imonoFeat));
        uniqfeatvals_this=unique(vals);
        if numel(uniqfeatvals_this)<=10 % treat as categorical variable
            feattypes(num2str(imonoFeat))='cat';
            feat_outlierlims(:,ff)=[min(uniqfeatvals_this);max(uniqfeatvals_this)];
        else % treat as continuous variable
            feattypes(num2str(imonoFeat))='cts';
            if strcmp( MCC_outlierdetection,'od_on')==1
                [lowlim,upplim]=outlier_limits(vals,3.5);
                vals=vals(vals>=lowlim);
                vals=vals(vals<=upplim);
                minx=min(vals);
                maxx=max(vals);
                feat_outlierlims(:,ff)=[lowlim;upplim];
           else % no outlier detection, use full global extrema
               minx=min(uniqfeatvals_this);
               maxx=max(uniqfeatvals_this);
               feat_outlierlims(:,ff)=[minx;maxx];
            end
            resolu=30;
            uniqfeatvals_this=(minx:(maxx-minx)/resolu:maxx)';
        end
        uniqfeatvals(num2str(imonoFeat))=uniqfeatvals_this;
        % get MCC results
        %if numel(MCC_mvals)==0
            nBootstraps=1;
            [MCC,allMCCs,m_vals] = calc_mcc_pmsvm_rbf(X,SVM_alphas,0,[],SVM_bias,SVM_y_train,SVM_X_train,SVM_kf,imonoFeat,boolIncreasing, nBootstraps,uniqfeatvals(num2str(imonoFeat)),'od_off',true);
        %else % provided with MCC results
        %    m_vals=MCC_mvals(:,:,ff);
            %[NC_dirn {-1,+1},NC_MT {0,-1}, Con_MT {0,1},Global_NC_Extent,Global_Con_Extent,NCR_NC_Extent,NCR_Con_Extent,NCHyperPlane,m {0 0.5 1}]
        %end
        nmtXs=X(  m_vals(:,9)~=1,:);
        nmt_mvals=m_vals(m_vals(:,9)~=1,:);
        X_base_pt_pool(num2str(imonoFeat))= nmtXs;
        X_base_pt_pool_std(num2str(imonoFeat))= (nmtXs-repmat(means(:,1)',[size(nmtXs,1) 1]))./repmat(stdevs(:,1)',[size(nmtXs,1) 1]);
        y_base_pt_pool(num2str(imonoFeat))=   SVM_y_train(m_vals(:,9)~=1,1);
        alphas_base_pt_pool(num2str(imonoFeat))=   SVM_alphas(m_vals(:,9)~=1,1);
        if size(nmtXs,1)>0
            ypred_base_pt_pool(num2str(imonoFeat))=  predict_svm_rbf(nmtXs,SVM_alphas,SVM_bias,SVM_y_train,SVM_X_train,SVM_kf);
        else
            ypred_base_pt_pool(num2str(imonoFeat))=[];
        end
        X_NMT_Info(num2str(imonoFeat))= nmt_mvals; 
    end

    % 30: ALLOCATE FEATURE BUDGET (number of base points per feature)
    % calculate number of constraints per basept
    num_constraints_per_basept=1;
    if strcmp(BasePtAnchor,'baseanchor_both')==1
        num_constraints_per_basept=num_constraints_per_basept+2;
    elseif strcmp(BasePtAnchor,'baseanchor_smart')==1
        num_constraints_per_basept=num_constraints_per_basept+1;
    end
    if strcmp(EndPtAnchor,'endanchor_both')==1
        num_constraints_per_basept=num_constraints_per_basept+2;
    elseif strcmp(EndPtAnchor,'endanchor_smart')==1
        num_constraints_per_basept=num_constraints_per_basept+1;
    end
    num_basept_equiv=ceil(numPairs/num_constraints_per_basept);
    if strcmp(FeatBudget,'equalbudget')==1
        basePtBudgets=zeros(size(constrFeats))+ceil(num_basept_equiv/numel(constrFeats));
    elseif strcmp(FeatBudget,'NMTpt_prop')==1
        tot_nmtpts=0;
        basePtBudgets=zeros(size(constrFeats));
        for ff=1:numel(constrFeats)
            imonoFeat=constrFeats(ff);
            nmt_info=X_NMT_Info(num2str(imonoFeat));
            cnt_nmtpts=numel(nmt_info(nmt_info(:,1)~=0));
            basePtBudgets(ff)=cnt_nmtpts;
            tot_nmtpts=tot_nmtpts+cnt_nmtpts;
        end
        if tot_nmtpts~=0 
            basePtBudgets=ceil(basePtBudgets/tot_nmtpts*double(num_basept_equiv));
        end
    end
    % 40: CREATE CONSTRAINTS 
    % create constraints  
    MCs=zeros(numPairs,size(Xcopy,2),2);
    iNextConstraintID=1;
    numduplicatesInARow=0;
    basepts_inds_used_all=containers.Map(); %zeros(0,numel(constrFeats));
    for ifeat=1:numel(constrFeats)
        basepts_inds_used_all(num2str(ifeat))=[];
    end
    ifeats_no_basepts=0;
    % construct traingle rand distribution
    prob=[0 1 1 1 0];
    x=[0 0.2 0.5  0.8 1];
    xi=0:0.05:1;
    pdf=interp1(x,prob,xi,'linear');
    pdf = pdf / sum(pdf);
    cdf = cumsum(pdf);
    [cdf, mask] = unique(cdf);
    xi=xi(mask);
    % create constraints
    while iNextConstraintID<=numPairs && ifeats_no_basepts < numel(constrFeats) && numduplicatesInARow<500
        shuf=randperm(numel(constrFeats));
        ifeats_no_basepts=0;
        for ifeat=shuf
            if numel(basepts_inds_used_all(num2str(ifeat)))==basePtBudgets(ifeat) % have aleady reached budget
                baseid=0;
                Xbase=[];
            else % still need to add base points
                feat=constrFeats(ifeat);
                X_base_pt_pool_thisfeat=X_base_pt_pool(num2str(feat));
                X_base_pt_pool_thisfeat_std=X_base_pt_pool_std(num2str(feat));
                y_base_pt_pool_thisfeat=y_base_pt_pool(num2str(feat));
                alphas_base_pt_pool_thisfeat=alphas_base_pt_pool(num2str(feat));
                ypred_base_pt_pool_thisfeat=ypred_base_pt_pool(num2str(feat));
                basepts_inds_used_feat=basepts_inds_used_all(num2str(ifeat));
                [nfeatparts,featparts]=get_partitions(Xcopy(:,feat),100,true,true);
                nfeatparts=iif(nfeatparts>=10,0,nfeatparts); % assign nfeatparts=0 to continuous variables, otherwise, featparts contains viable feature values
                
                if strcmp(BasePtSelnPriorCorr,'base_priorSVs')==1 % need to prioritise Support Vectors
                    pool_inds=1:size(X_base_pt_pool_thisfeat,1);
                    SVs_inds=pool_inds(alphas_base_pt_pool_thisfeat>max(alphas_base_pt_pool_thisfeat)*0.5);
                    basepts_inds_avail=setdiff(SVs_inds,basepts_inds_used_feat);
                    if numel(basepts_inds_avail)==0
                        basepts_inds_avail=setdiff(1:size(X_base_pt_pool_thisfeat,1),basepts_inds_used_feat);
                    end    
                else % no priority
                     basepts_inds_avail=setdiff(1:size(X_base_pt_pool_thisfeat,1),basepts_inds_used_feat);
                end
                %select base point at random
                if strcmp(BasePtSeln,'randseln')==1 % random
                    
                    if numel(basepts_inds_avail)==0 % none left to use!
                        if strcmp(BasePtsUnbounded,'ub_off')==1 % run out of base pts, stop
                            baseid=0;
                            Xbase=[];
                        else % 'ub_on': authorised to add more - add one at random
                            baseid=randi(size(X_base_pt_pool_thisfeat,1));
                            Xbase=X_base_pt_pool_thisfeat(baseid,:);
                        end
                    else % have some points available to use
                        baseid=basepts_inds_avail(randi(numel(basepts_inds_avail)));
                        Xbase=X_base_pt_pool_thisfeat(baseid,:);
                        basepts_inds_used_all(num2str(ifeat))=[basepts_inds_used_all(num2str(ifeat)) baseid];
                    end
                elseif strcmp(BasePtSeln,'maxminsep')==1 %  use greedy max-min-sep  
                    avail_pts=X_base_pt_pool_thisfeat(basepts_inds_avail,:);
                    avail_pts_std=X_base_pt_pool_thisfeat_std(basepts_inds_avail,:); %(avail_pts-repmat(means(:,1)',[size(avail_pts,1) 1]))./repmat(stdevs(:,1)',[size(avail_pts,1) 1]);
                    if numel(basepts_inds_avail)==0 % none left to use!
                        baseid=0;
                        Xbase=[];
                    else % have some points available to use
                        if numel(basepts_inds_used_feat)==0 % no previous points chosen. Use a random one to start.
                            newpt=randi(size(avail_pts,1));
                            basepts_inds_used_all(num2str(ifeat))=[newpt];
                        else % find max-min-sep pt
                            maxmindist=-9e9;
                            selpt=-9e9;
                            for p=1:numel(basepts_inds_avail)
                               diffs= X_base_pt_pool_thisfeat_std(basepts_inds_used_feat,:)-repmat(avail_pts_std(p,:),[numel(basepts_inds_used_feat) 1]);
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
                        baseid=newpt;
                        Xbase=X_base_pt_pool_thisfeat(baseid,:);
                    end
                end
            end
            % 40: CONSTRUCT CONSTRAINT AROUND BASE PT
            if baseid ==0 % we have no base poits to create a constraint on
                ifeats_no_basepts=ifeats_no_basepts+1;
            else % we have a base point to create a constraint on
                featVal=Xbase(1,feat);
                boolIncreasing=sum(decFeatures==feat)<=0;
                nmtinfo=X_NMT_Info(num2str(feat));
                NMTdirection=nmtinfo(baseid,1);
                %basepredy=predict_svm_rbf_real(Xbase,SVM_alphas,SVM_bias,SVM_y_train,SVM_X_train,SVM_kf);
                thisuniqfeatvals=uniqfeatvals(num2str(feat)); %thisuniqfeatvals(thisuniqfeatvals>featVal)
                feattype=feattypes(num2str(feat));
                % get constraint points %[NC_dirn {-1,+1},NC_MT {0,-1}, Con_MT {0,1},Global_NC_Extent,Global_Con_Extent,NCR_NC_Extent,NCR_Con_Extent,NCHyperPlane,m {0 0.5 1}]
                BASEPT_VAL=Xbase(1,feat);
                NC_HYPERPLANE=nmtinfo(baseid,8);
                NCR_NC_EXTENT=nmtinfo(baseid,6);
                NCR_Con_EXTENT=nmtinfo(baseid,7);
                GLOBAL_NC_EXTENT=nmtinfo(baseid,4);
                GLOBAL_Con_EXTENT=nmtinfo(baseid,5);
                % If this base pt has been used before, gather the existing
                % end pts to ensure they are not repeated
                % CONSTRUCT END PT
                if strcmp(EndPtSeln, 'randseln')==1
                    passiveNCRrange=sort(unique([NC_HYPERPLANE NCR_NC_EXTENT]),'ascend');
                    if numel(passiveNCRrange)==1
                        ENDPT_VAL = NC_HYPERPLANE;
                    else % have a range, pick a value
                        if strcmp(feattype,'cat')==1 % categorical
                            vals=thisuniqfeatvals(thisuniqfeatvals>=passiveNCRrange(1) & thisuniqfeatvals<passiveNCRrange(2));
                            ENDPT_VAL=vals(randi(numel(vals)));
                        else % continuous
                            % use a triangle uniform distribution (created above)
                            ENDPT_VAL=passiveNCRrange(1)+interp1(cdf,xi,rand)*(passiveNCRrange(2)-passiveNCRrange(1));
                        end
                    end
                elseif strcmp(EndPtSeln, 'closestRnminus1')==1
                    % NOT IMPLEMENTED
                end
                % CONSTRUCT ANCHORS
                BASE_ANCHOR=iif(GLOBAL_Con_EXTENT==NCR_Con_EXTENT,GLOBAL_Con_EXTENT,mean([BASEPT_VAL NCR_Con_EXTENT]));
                ENDPT_ANCHOR=iif(GLOBAL_NC_EXTENT==NCR_NC_EXTENT,GLOBAL_NC_EXTENT,mean([ENDPT_VAL NCR_NC_EXTENT]));
                BASE_ANCHOR_MID =mean([BASEPT_VAL NC_HYPERPLANE]);
                ENDPT_ANCHOR_MID=mean([ENDPT_VAL NC_HYPERPLANE]);
                % gather constraint points
                constraintpts=[BASEPT_VAL  ENDPT_VAL];
                if strcmp(BasePtAnchor, 'baseanchor_both')==1
                    constraintpts= [constraintpts BASE_ANCHOR BASE_ANCHOR_MID];
                elseif strcmp(BasePtAnchor, 'baseanchor_smart')==1
                    if abs(BASE_ANCHOR-BASEPT_VAL)>abs(BASE_ANCHOR_MID-BASEPT_VAL);
                        constraintpts= [constraintpts BASE_ANCHOR];
                    else
                        constraintpts= [constraintpts BASE_ANCHOR_MID];
                    end
                end
                if strcmp(EndPtAnchor, 'endanchor_both')==1
                    constraintpts= [constraintpts ENDPT_ANCHOR ENDPT_ANCHOR_MID];
                elseif strcmp(EndPtAnchor, 'endanchor_smart')==1
                    if abs(ENDPT_ANCHOR-ENDPT_VAL)>abs(ENDPT_ANCHOR_MID-ENDPT_VAL);
                        constraintpts= [constraintpts ENDPT_ANCHOR];
                    else
                        constraintpts= [constraintpts ENDPT_ANCHOR_MID];
                    end
                end
                constraintpts=sort(unique(constraintpts),'ascend');
                % construct constraints
                for md=1:numel(constraintpts)-1 %stdevs
                    % construct constraint points
                    constr1=Xbase;
                    constr2=Xbase;
                    if boolIncreasing
                         constr1(1,feat)= constraintpts(1,md);
                         constr2(1,feat)= constraintpts(1,md+1); 
                    else % decreasing
                         constr1(1,feat)= constraintpts(1,md+1);
                         constr2(1,feat)= constraintpts(1,md); 
                    end
                    % check whether this is a virtual duplicate of an
                    % existing constraint
                    if iNextConstraintID==1 ||  strcmp(DuplicatesCheck,'dupchk_off')==1 % haven't got any yet, or no duplicates check
                        isduplicate=0;
                    else % have some existing constraints, and dubchk_on
%                         isduplicate=0;
                        deviations_abs=abs([MCs(1:iNextConstraintID-1,:,1) MCs(1:iNextConstraintID-1,:,2)]- repmat([constr1 constr2],iNextConstraintID-1,1));
    %                     if strcmp(feattype,'cat')==1 % categorical
                        toocloseSDs_factor=0.2;
                        tooclose=deviations_abs<repmat(toocloseSDs_factor*[stdevs' stdevs'],size(deviations_abs,1),1);
                        [isduplicate,indx]=ismember(logical(ones(1,size(deviations_abs,2))),tooclose,'rows'); % ie if there are any existing rows with all values <0.1 stdevs, we have a duplicate
                        if isduplicate
                            numduplicatesInARow=numduplicatesInARow+1;
                        end
                    end
                    % add constraint
                    if ~isduplicate
                        numduplicatesInARow=0;
                        MCs(iNextConstraintID,:,1)=constr1;
                         MCs(iNextConstraintID,:,2)=constr2;
                         iNextConstraintID=iNextConstraintID+1;
                    end
                end
            end
            
        end
    end
    % trim additional points
    MCs=MCs(1:numPairs,:,:);
    % strip empty zeros rows if necessary
  keeptorow=0;
    for row=size(MCs,1):-1:1
        ifeatzeros=0;
       for ifeat=1:size(MCs,2)
          if  MCs(row,ifeat,1)==MCs(row,ifeat,2)
              ifeatzeros=ifeatzeros+1;
          end
       end
       if ifeatzeros==size(MCs,2)
           % continue
       else % have reached a real constraint
           keeptorow=row;
           break
       end
    end
    if keeptorow>0 
        MCs=MCs(1:keeptorow,:,:);
    end
end
