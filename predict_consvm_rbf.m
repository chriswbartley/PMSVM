function [classes scores] = predict_consvm_rbf(predXs,alphas,betas,MCs,b,y,X,kf)
%predict_consvm_rbf - predict a mc-svm model for set of inputs predXs
% DESCRIPTION:
% Predicts a constrained SVM using algorithm is as described in:
% Chen, C.C. & Li, S.T., 2014. Credit rating with a monotonicity-constrained 
% support vector machine model. Expert Systems with Applications, 41(16), pp.7235–7247.
%
% INPUTS:
%    predXs - TxP matrix of X vectors where class is to be predicted
%    alphas - Nx1 vector of solution for alpha Lagrangian multipliers (non-zero values correspond to support vectors)
%    betas - Mx1 vector of solution for beta Lagrangian multipliers (non-zero values correspond to support constraints)
%    MCs - MxPx2 matrix of M constraints where f(x_m')>=f(x_m) is guaranteed for 
%           x_m=MCs(m,:,1) and x_m'=MCs(m,:,2)
%    b - Bias term
%    y - Original Nx1 y vector used to train constained SVM
%    X - Original NxP X vector used by trained constained SVM
%    kf - RBF kernel factor
%
% OUTPUTS:
%    classes - Tx1 predicted classes based on predXs
%    scores - Tx1 predicted real value of constrained SVM (prior to converting to sign)
%
% Other m-files required: kernel_rbf
%

% Author: Chris Bartley
% University of Western Australia, School of Computer Science
% email address: christopher.bartley@research.uwa.edu.au
% Website: http://staffhome.ecm.uwa.edu.au/~19514733/
% Last revision: 30-March-2016

%------------- BEGIN CODE --------------
    n = size(y,1);
    m=size(MCs,1); 
    numPredXs=size(predXs,1);
    ys=zeros(numPredXs,1);
     scores=zeros(numPredXs,2);
%     % pre-filter on support vectos
%     tic
    sv_indexes=abs(alphas)>1e-8;
    alphas=alphas(sv_indexes);
    y=y(sv_indexes);
    X=X(sv_indexes,:);
    % pre-filter betas
    
    if numel(betas(betas>0))~=0
        sc_indexes=abs(betas)>1e-8;
        betas=betas(sc_indexes);
        MCs_filtered=zeros(numel(betas),size(MCs,2),2);
        MCs_filtered(:,:,1)=MCs(sc_indexes,:,1);
        MCs_filtered(:,:,2)=MCs(sc_indexes,:,2);
    end
    hasbetas=numel(betas(betas>0))>0;
    % calculate results for each prediction point
    for ix=1:numPredXs
        res=sum(alphas.*y.*kernel_rbf(X,repmat(predXs(ix,:),size(X,1),1),kf));
        if hasbetas==1
            res=res+ sum(betas.*(kernel_rbf(MCs_filtered(:,:,2),repmat(predXs(ix,:),numel(betas),1),kf)-kernel_rbf(MCs_filtered(:,:,1),repmat(predXs(ix,:),numel(betas),1),kf)));
        end
        ys(ix,1)=iif((res+b)>0,+1,-1);
        scores(ix,:)=(res+b)*[-1 1];
    end
% toc
%    tic 
% % old, slow method
%      
%     for ix=1:numPredXs
%         res=0;
%         for i=1:n
%             res=res+alphas(i)*y(i)*kernel_rbf(X(i,:),predXs(ix,:),kf);
%         end
%         for i = 1:m % MC terms
%            res=res+ betas(i)*(kernel_rbf(MCs(i,:,2),predXs(ix,:),kf)-kernel_rbf(MCs(i,:,1),predXs(ix,:),kf));
%         end
%         ys(ix,1)=iif((res+b)>0,1,-1);
%         scores(ix,:)=(res+b)*[-1 1];
%     end
% toc
    classes=iif(numPredXs==1,ys(1,1),ys);
    scores=iif(numPredXs==1,scores(1,1),scores);
end

