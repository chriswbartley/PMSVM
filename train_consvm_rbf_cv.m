function [loss,ypred_all,cm_by_fold,  minEigs_by_fold]  = train_consvm_rbf_cv(y,X,MCs,bc,kf, cvpartitions)
%train_consvm_rbf_cv - Solves train_consvm_rbf for a given set of CV partitions
%
% INPUTS:
%    y - Nx1 class vector (+1,-1)
%    X - NxP training data
%    MCs - MxPx2 matrix of M constraints where f(x_m')>=f(x_m) is guaranteed for 
%    x_m=MCs(m,:,1) and x_m'=MCs(m,:,2)
%    bc - SVM box constraint
%    kf - RBF kernel factor
%    cvpartitions - either Nx1 vector of CV fold numbers, or matlab's
%    CVPartition object
%
% OUTPUTS:
%    loss - Fx1 MCR values for F-fold CV partition results
%    ypred_all - Nx1 predicted y class for when fold containing y was left out
%    cm_by_fold - Fx4 confusion matrix for each fold results, where cols
%    are TP TN FP FN 
%    minEigs_by_fold - Fx1 vector of minimmum solution matrix eigenvalue.
%    If minEig<0, 2*abs(minEig) was added to solution matrix diagonal (Tikhonov 
%    regularisation) prior to solution to ensure convexity.
%
% Other m-files required: train_consvm_rbf, kernel_rbf

% Author: Chris Bartley
% University of Western Australia, School of Computer Science
% email address: christopher.bartley@research.uwa.edu.au
% Website: http://staffhome.ecm.uwa.edu.au/~19514733/
% Last revision: 30-March-2016

%------------- BEGIN CODE --------------
    % prelims
    n=size(X,1);
    ypred_all=zeros(n,1);
    neg_y=min(y);
    % convert CV partiition to index array
    if isa(cvpartitions,'cvpartition')
        cvpartArray=zeros(n,1);
        for i=1:cvpartitions.NumTestSets
           cvpartArray( cvpartitions.test(i))=i;
        end
        numCVfolds=cvpartitions.NumTestSets;
    else
        cvpartArray=cvpartitions;
        numCVfolds=max(cvpartArray);
    end
    loss = zeros(numCVfolds,1);
    cm_by_fold=zeros(numCVfolds,4); % TP TN FP FN 
    FMR_by_fold=zeros(numCVfolds,1);
    MCC_by_fold=zeros(numCVfolds,1);
    minEigs_by_fold=zeros(numCVfolds,1);
    % run CV partitions
    for i =1:numCVfolds %cvpartitions.NumTestSets
        ind_train = cvpartArray~=i;%cvpartitions.training(i);
        ind_test= cvpartArray==i;%cvpartitions.test(i);
        ntrain=size(ind_train(ind_train>0),1);
        ntest=size(ind_test(ind_test>0),1);
        % calculate alphas
        %MCs=getMCsWrapper(X(ind_train,:),MCs_type, numMCs,incrFeatures,decFeatures,y(ind_train));
        %MCs=generateMCpairsCB1(X(ind_train,:), numMCs,incrFeatures,decFeatures);
        [alphas,betas, y_pred,bias, H,minEig] = train_consvm_rbf(y(ind_train),X(ind_train,:),bc,kf, X(ind_test,:),MCs);
        % calculate confusion matrix
        cm= confusionmat(y(ind_test),y_pred);
        if numel(cm)==1 % if accuracy is 100%, confusionmat can return a signle element 
            if y_pred(1,1)==neg_y % all are TN
                cm=[cm 0;0 0];
            else % all are TP
                cm=[0 0;0 cm];
            end
        end
        cm_by_fold(i,:)=[cm(2,2) cm(1,1) cm(1,2) cm(2,1)];% TP TN FP FN 

        minEigs_by_fold(i)=minEig;
        % calculate other metrics
        ypred_all(ind_test)=y_pred;
        ydiff = y_pred-y(ind_test);
        incorrect=size(ydiff(ydiff~=0),1);
        loss(i) = incorrect/ntest; 
    end
    
end
