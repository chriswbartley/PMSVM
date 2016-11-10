function [summary,movements] = calc_mcc_interp_pmsvm_rbf(predXs,alphas,betas,MCs,b,y,X,kf,imonoFeat, uniqfeatvals,outlierdetection)
%calc_mcc_interp_pmsvm_rbf - Estimates the effect of a given feature on a
%   PM-SVM model's output class.
% DESCRIPTION:
%   This is an implementation of the summary monotonicity effects of a
%   feature at all datapoints built on the partial monotonicity framework
%   proposed in Bartley et al. 2016 'Effective Knowledge Integration in 
%   Support Vector Machines for Improved Accuracy' (submitted to
%   ECMLPKDD2016).In particular it enables the disagreement statistic
%   (Equation 10) to be calculated.
% 
%   For a given training data and PM-SVM model this classes each data point
%   as one of: (a) Not Changing in either direction (b) Monotone Increasing 
%   (in one or both directions) (c) Monotone Decreasing (in one or both 
%   directions) or (d) Monotone Increasing and Decreasing (one in each 
%   direction). When used with the unconstrained SVM model, this information 
%   can be used to calculate the disagreement statistic (Equation 10). The
%   information can also be used to summarise the impact of a feature on
%   the output class over the input space (weighted by the joint pdf).
%
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
%   summary - 4x1 matrix giving the percentage (of the T test points) where
%       the function is: [NoChangeInEitherDirn , MonotoneIncreasing (one
%       or both directions),MonotoneDecreasing (one or both directions),
%       MonotoneIncrDec (one direction each)]. For an (increasing) feature,
%       this can be used to calculate the disagreement percentage (Equation
%       10 in paper): 
%       dis=(MonotoneDecreasing+0.5*MonotoneIncrDec)/(MonotoneIncreasing+MonotoneDecreasing+MonotoneIncrDec)
%   movements - Tx2 matrix showing directino of change in f(x) in
%       decreasing (column 1) and increasing (column 2) feature directions.
%       Values are -1 (first change is a decrease in f(x), 0 (no change), or +1
%       (first change is an increase in f(x).
%
% Other m files needed: predict_consvm_rbf, outlier_limits, boxcox
% See also: train_consvm_rbf

% Author: Chris Bartley
% University of Western Australia, School of Computer Science
% email address: christopher.bartley@research.uwa.edu.au
% Website: http://staffhome.ecm.uwa.edu.au/~19514733/
% Last revision: 30-March-2016
    n = size(y,1);
    m=size(MCs,1); 
    numXs=size(predXs,1);
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
               maxx=max(uniqfeatvals);
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
    movements=zeros(size(predXs,1),2);
    %nonmonopts=zeros(0,3);
    for i=1:size(predXs,1)
        % get the current value at x_i
        currx_i_p=predXs(i,imonoFeat);
        % get all predictions for uniq vals
        predictedys_uniqvals=predict_consvm_rbf_featvars(predXs(i,:),alphas,betas,MCs,b,y,X,kf,imonoFeat,uniqfeatvals);
        % get this prediction for curr X
        if m==0
            origy=predict_svm_rbf(predXs(i,:),alphas,b,y,X,kf);
        else
            origy=predict_consvm_rbf(predXs(i,:),alphas,betas,MCs,b,y,X,kf);
        end
        % look at prior values then post values
        for searchdirn=[-1 +1]
            % search in potential non-monotone direction first
            if searchdirn==1
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
                        if searchdirn==-1 % looking prior
                            movements(i,1)=iif(predy_prior<lasty,-1,+1);
                        else
                            movements(i,2)=iif(predy_prior<lasty,-1,+1);
                        end
                        break
                    end
                end

            end
        end
      
    end
    % calculate summary
      numXs=size(predXs,1);
      summary=[0. 0. 0. 0.];
      summary(1)=sum(movements(:,1)==0 & movements(:,2)==0 )/numXs; % no change
      summary(2)=sum((movements(:,1)<0 & movements(:,2)==0) | (movements(:,1)==0 & movements(:,2)>0)| (movements(:,1)<0 & movements(:,2)>0))/numXs; % increasing
      summary(3)=sum((movements(:,1)>0 & movements(:,2)==0) | (movements(:,1)==0 & movements(:,2)<0)| (movements(:,1)>0 & movements(:,2)<0))/numXs; % decreasing
      summary(4)=sum((movements(:,1)>0 & movements(:,2)>0) | (movements(:,1)<0 & movements(:,2)<0))/numXs; % decreasing
      check=sum(summary);
      if (abs(check-1.))>1e-4
         disp ('***************** MCC INTERP CHECK IS WRONG *****************'); 
      end
end

