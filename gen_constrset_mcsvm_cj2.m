function MCs = gen_constrset_mcsvm_cj2(X, numPairs,incrFeatures,decFeatures,params)
%gen_constrset_mcsvm_cj2 - Generates a set of conjunctive 
%constraints for use with constrained SVM. Based on Li & Chen 2014
% DESCRIPTION:
%   This generates a MxPx2 matrix which represents a set of M constraints to be 
%   used with the constraned SVM algorithm (train_consvm_rbf). Each constraint is
%   the pair of points (x_m,x_m') where x_m=MCs(m,:,1) and x_m'=MCs(m,:,2).
%   train_consvm_rbf() can then be used to train an SVM for which f(x_m')>=f(x_m) 
%   is guaranteed for this set of constraints.
%   This algorithm is used for CJ2 contraints as described in Bartley 
%   et al. 2016 'Effective Knowledge Integration in Support Vector Machines 
%   for Improved Accuracy' (submitted to ECMLPKDD2016).
%
%   This technique is an implementation of that proposed in
%    Li, S.-T. & Chen, C.-C., 2014. A Regularized Monotonic Fuzzy Support Vector 
%    Machine for Data Mining with Prior Knowledge. Fuzzy Systems, IEEE Trans, PP(99).
%   This paper did not describe what to do with the unconstrained features, so 
%   to influence the densest part of the input space we set these to the 
%   data-point closest (in Euclidean distance) to the centroid as
%   determined by a 1-cluster normalised k-means analysis.
%
% INPUTS:
%    X - NxP matrix of training data
%    numPairs - the number of constraints to be created (M)
%    incrFeatures - p_inc scalar indicating which features (p_inc subset of
%    (1..P)) are INCREASING monotone
%    decFeatures - p_dec scalar indicating which features (p_dec subset of
%    (1..P)) are DECREASING monotone
%    params - NOT USED
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
%   MCs=gen_constrset_mcsvm_cj2(Xtrain, 500,[2 5],[3 7],{}) 
%   [alphas,betas, y_pred,bias,H, minEig] = train_consvm_rbf(ytrain,Xtrain,bc,kf,Xtest, MCs)
%
% Other m-files required: get_partitions
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
    constrFeats=[incrFeatures decFeatures];
    myrandstrm= RandStream('mcg16807','seed','shuffle');
    % reverse decreasing features and correct at the end of the function
    for i=1:numel(decFeatures)
       Xcopy(:,decFeatures(i))= -Xcopy(:,decFeatures(i));
    end
    if numel(constrFeats)==size(X,2) % all features constraineed, no need to worry about unconstrained features
        basept=zeros(1,size(X,2));
    else % work out a value for the unconstrained features
        % calculate the centroid to provide values for the unconstrained
        % features
        % normalise the data
        X_norm=X;
        for i=1:size(X,2)
            v=X_norm(:,i);
            if ~all(v == v(1)) % ensure all values for this feature are not the same! else stdev=0 and get problems
                X_norm(:,i)=(X(:,i)-mean(X(:,i)))/std(X(:,i)); 
            end
        end
        % find optimal k-means clustering
        eva = evalclusters(X_norm,'kmeans','gap','KList',[1]);
        %OptimalK=eva.OptimalK;
        %clusters=eva.OptimalY;
        dist=eva.StdLogW;
        [minval,mini]=min(dist);
        basept=Xcopy(mini,:);
    end
    % get partitions
    partitions=zeros(numPairs,numel(constrFeats),2)-99;
    for ifeat=1:numel(constrFeats)
        feat=constrFeats(ifeat);
        [nparts,parts]=get_partitions(Xcopy(:,feat),numPairs,true,false); 
        for ipart=1:nparts %numPairs %nparts
            looping_ipart=ipart; %1+mod(ipart-1,nparts);
           partitions(ipart,ifeat,1)=parts(looping_ipart);
           partitions(ipart,ifeat,2)=parts(looping_ipart+1);
        end
        if nparts<numPairs % need to add padding to end
           for ipart=nparts+1:numPairs
               partitions(ipart,ifeat,1)=parts(nparts+1);
               partitions(ipart,ifeat,2)=parts(nparts+1);
           end
        end
    end
     MCs=zeros(numPairs,size(Xcopy,2),2);
     for i=1:numPairs
         MCs(i,:,1)=basept;
         MCs(i,:,2)=basept;
         %for part=1:size(partitions,1);
            % correct constrained features
            for ifeat=1:numel(constrFeats)
                feat=constrFeats(ifeat);
                MCs(i,feat,1)=partitions(i,ifeat,1);
                MCs(i,feat,2)=partitions(i,ifeat,2);
            end
       % end
     end
    
    % re-reverse decreasing features
    for i=1:numel(decFeatures)
       MCs(:,decFeatures(i),1)= -MCs(:,decFeatures(i),1);
       MCs(:,decFeatures(i),2)= -MCs(:,decFeatures(i),2);
    end
    % trim additional points
    MCs=MCs(1:numPairs,:,:);
end
