function MCs = gen_constrset_mcsvm_cj1(X, numPairs,incrFeatures,decFeatures,params)
%gen_constrset_mcsvm_cj1 - Generates a set of conjunctive 
%constraints for use with constrained SVM. Based on Chen & Li 2014
% DESCRIPTION:
%   This generates a MxPx2 matrix which represents a set of M constraints to be 
%   used with the constraned SVM algorithm (train_consvm_rbf). Each constraint is
%   the pair of points (x_m,x_m') where x_m=MCs(m,:,1) and x_m'=MCs(m,:,2).
%   train_consvm_rbf() can then be used to train an SVM for which f(x_m')>=f(x_m) 
%   is guaranteed for this set of constraints.
%   This algorithm is used for CJ1 contraints as described in Bartley 
%   et al. 2016 'Effective Knowledge Integration in Support Vector Machines 
%   for Improved Accuracy' (submitted to ECMLPKDD2016).
%
%   This technique is an implementation of that proposed in
%   Chen, C.C. & Li, S.T., 2014. Credit rating with a monotonicity-constrained 
%   support vector machine model. Expert Systems with Applications, 41(16), pp.7235–7247.
%   This paper did not describe what to do with the unconstrained features, so 
%   we used the values of one of the nCandidates data-points selected at random.
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
%               each contraint. '5' was used for the ECMLPKDD paper.
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
%   MCs=gen_constrset_mcsvm_cj2(Xtrain, 500,[2 5],[3 7],{'5'}) 
%   [alphas,betas, y_pred,bias,H, minEig] = train_consvm_rbf(ytrain,Xtrain,bc,kf,Xtest, MCs)
%
% Other m-files required: NONE
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
    constrFeats=[incrFeatures decFeatures];
    myrandstrm= RandStream('mcg16807','seed','shuffle');
    % reverse decreasing features and correct at the end of the function
    for i=1:numel(decFeatures)
       Xcopy(:,decFeatures(i))= -Xcopy(:,decFeatures(i));
    end
    % create constraints  
    MCs=zeros(numPairs,size(Xcopy,2),2);
    for i=1:numPairs
        Xcandidate=datasample(myrandstrm,Xcopy,nCandidates,'Replace',false);
        % select one data point at random from the candidates to be the
        % base data point for this constraint
        baseid=randi(nCandidates);
        Xbase=Xcandidate(baseid,:);
        % add default to new constraint points
        MCs(i,:,1)=Xbase;
        MCs(i,:,2)=Xbase;
        % correct constrained features
        for j= [decFeatures incrFeatures]
            MCs(i,j,1)=min(Xcandidate(:,j));
            MCs(i,j,2)=max(Xcandidate(:,j));
        end
    end
    % re-reverse decreasing features
    for i=1:numel(decFeatures)
       MCs(:,decFeatures(i),1)= -MCs(:,decFeatures(i),1);
       MCs(:,decFeatures(i),2)= -MCs(:,decFeatures(i),2);
    end
end
