function [MCC,allMCCs,m_vals] = calc_mcc_pmsvm_rbf(predXs,alphas,betas,MCs,b,y,X,kf,imonoFeat,boolIncreasing, nBootstraps,uniqfeatvals,outlierdetection,calculateNCR_ConExtent)
%calc_mcc_pmsvm_rbf - Calculates Monotonicity Compliance for given PM-SVM
%model and X data and feature imonoFeat. Typically X partition should be the TEST partition for
%accurate measurement.
% DESCRIPTION:
%   This is an implementation of the Monotonicity Compliance (MCC) measure
%   proposed in  Bartley et al. 2016 'Effective Knowledge Integration in 
%   Support Vector Machines for Improved Accuracy' (submitted to ECMLPKDD2016).
%   See this paper for a thorough definition.
%
%   NOTE: this implementation is optimised for BINARY output class
%   problems, because in that case you only need to search in one direction
%   for possible monotonicity. For multi-class or continuous output
%   problems an implementation would always need to search in both
%   directions.
%
% INPUTS:
%    predXs - TxP matrix of test data on which to assess monotonicity.
%    alphas - Nx1 vector of solution for alpha Lagrangian multipliers (non-zero values correspond to support vectors)
%    betas - Mx1 vector of solution for beta Lagrangian multipliers (non-zero values correspond to support constraints)
%    MCs - MxPx2 matrix of M constraints (used for model PM-SVM model
%    creation)
%    b - Bias term
%    y - Original Nx1 y vector used to train constained SVM
%    X - Original NxP X vector used by trained constained SVM
%    kf - RBF kernel factor
%    imonoFeat - feature for which MCC is to be calculated
%    boolIncreasing - 1 for an increasing feature, 0 for a decreasing
%    feature
%    nBootstraps - number of bootstrap samples, say 10,000
%    uniqfeatvals - An increasing vector of feature values that will be
%       used as the feature values where the function is tested. Use [] for
%       this to be determined automatically.
%    outlierdetection - 'od_on' will use outlier detection when determining 
%       the limits of each feature test (ie
%       global min and max will not necessarily be used, based on outlier 
%       detection usinga boxcox normalised 3 stdev definition for outlier.
%    calculateNCR_ConExtent - For a binary class problem, to assess 
%       monotonicity it is only necessary to check for non-monotonicity in 
%       one direction (if f(x)=+1, we only need to check in the increasing 
%       direction, and vice versa for f(x)=-1.). Thus we will not hunt in
%       the other direction unless calculateNCR_ConExtent=1. (and note that
%       unless calculateNCR_ConExtent=1, NCR_Con_Extent will not be
%       accurate).
%               
% OUTPUTS:
%   MCC - Estimated MCC for imonofeat on predXs (scalar)
%   allMCCs - (nBootstraps x 1) vector of MCC values for all bootstraps
%   m_vals - Tx9 matrix containing all test information for each test
%       point. The columns correspond to:
%         col 1- Non-conforming direction: +1 (increasing feat val) or -1
%         (decreasing)
%         col 2- Non-conforming direction change:-1 (non-monotone) or 0 - (no change)
%         col 3- Conforming direction change: for other direction, value is 0 (no change) or 1 (monotone step)
%         col 4- Global_NC_Extent: maximum (or minimmum) value of feature
%         tested in NC direction. This will equal the feature max/min,
%         unless outlierdetection is on and some outlier feat vals are
%         detected.
%         col 5- Global_Con_Extent: Same as above but in the conforming
%         direction.
%         col 6- NCR_NC_Extent: NMR extent on far side of Active NMR (See paper fig 2)
%         col 7- NCR_Con_Extent: NMR extent on far side of Passive NMR. It is not 
%           necessary to know this to calculate monotonicity, so this value 
%           will not be accurate unless calculateNCR_ConExtent=1.
%         col 8- NCHyperPlane: location of nonconforming hyperplane (if m<>1)
%         col 9- m: m_ci(x) value from Equation 7 of paper. This is 0 (non-monotone) or 1 (monotone/no change)
%           note that for binary output this cannot equal 0.5.
%
% Other m files needed: predict_consvm_rbf, outlier_limits, boxcox
% See also: train_consvm_rbf

% Author: Chris Bartley
% University of Western Australia, School of Computer Science
% email address: christopher.bartley@research.uwa.edu.au
% Website: http://staffhome.ecm.uwa.edu.au/~19514733/
% Last revision: 30-March-2016

%------------- BEGIN CODE --------------
    n = size(y,1);
    m=size(MCs,1); 
    numXs=size(predXs,1);
    allMCCs=zeros(nBootstraps,1);
    llNoEffects=zeros(nBootstraps,1);
    % calculate uniqfeatvals, if required
    if numel(uniqfeatvals)==0 % need to calculate uniqfeatvals
        vals=sort(predXs(:,imonoFeat));
        uniqfeatvals=unique(vals);
        if numel(uniqfeatvals)>10 % treat as continuous variable
            if strcmp( outlierdetection,'od_on')==1
                [lowlim,upplim]=outlier_limits(vals);
                vals=vals(vals>=lowlim);
                vals=vals(vals<=upplim);
                minx=min(vals);
                maxx=max(vals);
           else % no outlier detection, use full global extrema
               minx=min(uniqfeatvals);
               maxx=max(uniqfeatvals)
            end
            resolu=30;
            uniqfeatvals=(minx:(maxx-minx)/resolu:maxx)';
        end
    end
    if size(uniqfeatvals,2)>size(uniqfeatvals,1)
        uniqfeatvals=uniqfeatvals';
    end
    % get global extents
    global_max=max(uniqfeatvals);
    global_min=min(uniqfeatvals);
    
    % for each datapoint, calculate m(x) monotonicity function (0,0.5,1)
    m_func=zeros(size(predXs,1),9)-99e5;
    no_effect_func=zeros(n,1);
    %nonmonopts=zeros(0,3);
    for i=1:size(predXs,1)
        % get the current value at x_i
        currx_i_p=predXs(i,imonoFeat);
        % get all predictions for uniq vals
        % option one - feat variants
%         tic
%         predictedys_uniqvals=predict_consvm_rbf_featvars(predXs(i,:),alphas,betas,MCs,b,y,X,kf,imonoFeat,uniqfeatvals);
%         toc
        % option 2: faster prediction tecchnique (0.02 vs 0.15 sec)
%         tic
        predXvariants=repmat(predXs(i,:),numel(uniqfeatvals),1);
        predXvariants(:,imonoFeat)=uniqfeatvals;
        predictedys_uniqvals=predict_consvm_rbf(predXvariants,alphas,betas,MCs,b,y,X,kf);
%         toc
        % get this prediction for curr X
        if m==0
            origy=predict_svm_rbf(predXs(i,:),alphas,b,y,X,kf);
        else
            origy=predict_consvm_rbf(predXs(i,:),alphas,betas,MCs,b,y,X,kf);
        end
        % get potential non-monotone direction
        if origy==1
           nmt_searchdirn=iif(boolIncreasing,1,-1); 
        else % origy==-1
           nmt_searchdirn=iif(boolIncreasing,-1,1); 
        end
        % search in potential non-monotone direction first
        if nmt_searchdirn==1
            nextxvals=sort(uniqfeatvals(find(uniqfeatvals>currx_i_p)),'ascend');
        else % search backwards
            nextxvals=sort(uniqfeatvals(find(uniqfeatvals<currx_i_p)),'descend');
        end 
        if numel(nextxvals)==0
            k=0;
            hyperplanes_encountered=0;
        else % have some prior values to test
            ks_all=size(nextxvals,1);
            hyperplanes_encountered=0;
            lasty=origy;
            for k=1:ks_all  
               x_k=predXs(i,:);
               prior_feat_val=nextxvals(k,1);
                x_k(1,imonoFeat)=prior_feat_val;
                [r,c]=find(uniqfeatvals==prior_feat_val);
                predy_prior=predictedys_uniqvals(r,1);
                if predy_prior~=lasty
                    if hyperplanes_encountered==0 % found the problem HP
                        hyperplanes_encountered=1;
                        php=prior_feat_val;
                        lasty=predy_prior;
                        if ~calculateNCR_ConExtent % don't continue
                            NCR_NC_Extent=-99e5;
                            break
                        end
                    else % found the NCR Problem Extent, stop looking
                        NCR_NC_Extent = prior_feat_val;
                        hyperplanes_encountered=2;
                        break
                    end 
                end
            end
            
        end
        %disp([num2str(numel(nextxvals)) ' ' num2str(k) ' ' num2str(hyperplanes_encountered)]);
        if k==0 || hyperplanes_encountered==0 % at a 'beginning' section, or no change to beginning
            mono_nmtdirn=0;
            php=-99e5; % no PHP
            NCR_NC_Extent=-99e5;
        else %not at a beginning section, there has been a change and therefore a NMT hyperplane
            mono_nmtdirn=-1; %iif(boolIncreasing,iif(predy_prior<origy,1,-1),iif(predy_prior>origy,1,-1));
            % php will have been set
            if hyperplanes_encountered<2
                NCR_NC_Extent=prior_feat_val; % last val searched
            end
        end
        if nmt_searchdirn==1
            nextxvals=sort(uniqfeatvals(find(uniqfeatvals<currx_i_p)),'descend'); 
        else % search backwards
            nextxvals=sort(uniqfeatvals(find(uniqfeatvals>currx_i_p)),'ascend');
        end 
        if numel(nextxvals)==0
            k=0;
        else % have some prior values to test
            ks_all=size(nextxvals,1);
            k=1;
            x_k=predXs(i,:);
            after_feat_val=nextxvals(k,1);
            x_k(1,imonoFeat)=after_feat_val;
            [r,c]=find(uniqfeatvals==after_feat_val);
            predy_after=predictedys_uniqvals(r,1);
            while predy_after==origy && k<=ks_all-1
               k=k+1; 
               x_k=predXs(i,:);
               after_feat_val=nextxvals(k,1);
                x_k(1,imonoFeat)=after_feat_val;
                [r,c]=find(uniqfeatvals==after_feat_val);
            predy_after=predictedys_uniqvals(r,1);
%                    if m==0
%                         predy_after=predict_svm_rbf(x_k,alphas,b,y,X,kf);
%                     else
%                         predy_after=predict_consvm_rbf(x_k,alphas,betas,MCs,b,y,X,kf);
%                     end
            end
        end
        if k==0 || predy_after==origy % at a 'beginning' section, or finished with no change
            mono_mtdirn=0;
            NCR_Con_Extent=iif(nmt_searchdirn==1,global_min,global_max);
        else %not at a beginning section, check monotonicity of first kick
            NCR_Con_Extent=after_feat_val;
            mono_mtdirn=1; %iif(boolIncreasing,iif(predy_after>origy,1,-1),iif(predy_after<origy,1,-1));
        end
        % work out final result for m [NC_dirn {-1,+1},NC_MT {0,-1}, Con_MT {0,1},Global_NC_Extent,Global_Con_Extent,NCR_NC_Extent,NCR_Con_Extent,NCHyperPlane,m {0 0.5 1}]
        m_func(i,1)=nmt_searchdirn;
        m_func(i,2)=mono_nmtdirn;
        m_func(i,3)=mono_mtdirn;
         m_func(i,4)=iif(nmt_searchdirn==1,global_max,global_min);
          m_func(i,5)=iif(nmt_searchdirn==1,global_min,global_max);
           m_func(i,6)=NCR_NC_Extent;
            m_func(i,7)=NCR_Con_Extent;
             m_func(i,8)=php;
        if mono_mtdirn==0 && mono_nmtdirn==0  % does not change for all values.
            m_func(i,9)=1;
            no_effect_func(i)=1;
        else
            if mono_nmtdirn==0 && mono_mtdirn>=0
                m_func(i,9)=1;
            elseif -mono_nmtdirn ==mono_mtdirn
                m_func(i,9)=0.5;
            else
                m_func(i,9)=0;
            end
            %m_func(i,1)=(iif(mono_prior>=0,mono_prior,0)+iif(mono_after>=0,mono_after,0))/iif(mono_after>=0 && mono_prior>=0,2,1);           
            no_effect_func(i)=0;
        end
    end
    
    % calc bootstraps
    for b=1:nBootstraps
        % get bootstrap sample of predXs
        samp = randsample(numXs,numXs,true);
        derivs=m_func(samp,9); %TEMP OVERRIDE predXs
        noeffs=no_effect_func(samp);
        % calculate MCC from derivatives
        allMCCs(b)=sum(derivs)/numXs;
    end
    MCC=mean(allMCCs(:,1));
    m_vals=m_func;
end
