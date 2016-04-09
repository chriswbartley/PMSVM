function [alphas,betas, y_pred,bias,H, minEig] = train_consvm_rbf(ytrain,Xtrain,bc,kf,Xtest, MCs)
%train_consvm_rbf - trains a C-SVM binary classifier with monotone constraints.
%
% DESCRIPTION:
% In essence the resulting SVM classifier will respect f(x')>=f(x) for all
% pairs of constraint points (x,x') in constraint set MCs.
% Core algorithm is as described in:
% Chen, C.C. & Li, S.T., 2014. Credit rating with a monotonicity-constrained 
% support vector machine model. Expert Systems with Applications, 41(16), pp.7235–7247.
%
% INPUTS:
%    ytrain - Nx1 vector of binary class y (-1 / +1)
%    Xtrain - NxP matrix of P-dimensional X data. 
%    bc - Box constraint for C-SVM
%    kf - Kernel Factor for RBF kernel.
%    Xtest - TxP test dataset of length T (may use Xtrain if no separate
%    test partition)
%    MCs - MxPx2 matrix of M constraints where f(x_m')>=f(x_m) is guaranteed for 
%    x_m=MCs(m,:,1) and x_m'=MCs(m,:,2). You can create these constraintsets for
%    PM-SVM using gen_constrset_pmsvm_nc or gen_constrset_pmsvm_adaptive.
%
% OUTPUTS:
%    alphas - Nx1 vector of solution for alpha Lagrangian multipliers (non-zero values correspond to support vectors)
%    betas - Mx1 vector of solution for beta Lagrangian multipliers (non-zero values correspond to support constraints)
%    y_pred - Tx1 vector of predicted y values for Xtest
%    bias - SVM bias scalar
%    H - Solution matrix solved by quadprog
%    minEig - minimum eigenvalue of solution matrix - if minEig<0, and diagonal
%    correction of +2*abs(minEig) is included in H to guarantee problem
%    convexity (Tikhonov regularisation).
%
% EXAMPLE:
%   To create 500 constraints, with increasing features [2 5] and
%   decreasing features [3 7], with no mid pts but with greedy mms and duplicate 
%   detection, and then solve the PM-SVM model:
%   MCs=gen_constrset_pmsvm_nc(Xtrain, 500,[2 5],[3 7],{'5','0','1','1'}) 
%   [alphas,betas, y_pred,bias,H, minEig] = train_consvm_rbf(ytrain,Xtrain,bc,kf,Xtest, MCs)
%
% Other m-files required: kernel_rbf
%

% Author: Chris Bartley
% University of Western Australia, School of Computer Science
% email address: christopher.bartley@research.uwa.edu.au
% Website: http://staffhome.ecm.uwa.edu.au/~19514733/
% Last revision: 30-March-2016

    % prelimsminE
    n=size(Xtrain,1);
    m=size(MCs,1);
    % hessian matrix
    G11=zeros(n,n);
    for row=1:n
        for col=1:n
            %Xd=Xtrain(row,:)-Xtrain(col,:);
            G11(row,col)=ytrain(row)*ytrain(col)*kernel_rbf(Xtrain(row,:),Xtrain(col,:),kf); %(-Xd*Xd'/kf^2);
        end
    end
    G12=zeros(n,m);
    for i=1:n
        for j=1:m
            G12(i,j)=ytrain(i)*(kernel_rbf(MCs(j,:,2),Xtrain(i,:),kf)-kernel_rbf(MCs(j,:,1),Xtrain(i,:),kf)); %(-Xd*Xd'/kf^2);
        end
    end
    G22=zeros(m,m);
    for i=1:m
        for j=1:m
            G22(i,j)=kernel_rbf(MCs(i,:,2),MCs(j,:,2),kf)  +  kernel_rbf(MCs(i,:,1),MCs(j,:,1),kf) - kernel_rbf(MCs(i,:,2),MCs(j,:,1),kf) - kernel_rbf(MCs(i,:,1),MCs(j,:,2),kf); %(-Xd*Xd'/kf^2);
        end
    end
    % lazy way to correct small asymmetries in G22 (1e-22 differences!)
    % Otherwise quadprog gives warning
    G22=(G22+G22')/2;
    % create matlab quadprog variables
    G=[G11 G12; G12' G22];
    eigenvaluesH=eig(G);
    minEig=min(eigenvaluesH);
    f=-1*ones(n+m,1);
    f(n+1:n+m)=0.;
    A=[];
    b=[];
    Aeq=[ytrain(:,1); zeros(m,1)]';
    beq=0.0; %zeros(1,1);
    lb=zeros(n+m,1);
    ub=bc*ones(n+m,1);
    ub(n+1:n+m)=1e10; % effectively remove upper bound on betas
    % solve for weights
    if minEig<0
        increment=2*abs(minEig);
    else
        increment=0;
    end %iif(minEig<0,2*abs(minEig),0); % tikhonov regularisation (if required)
    H=G+increment*eye(size(G)); 
    alphasbetas=quadprog(H,f,A,b,Aeq,beq,lb,ub,0,optimset('Display','off'));
    alphas=alphasbetas(1:n,:);
    betas=alphasbetas(n+1:n+m,:);
    % Calculate bias 'bias'. For support vectors 0<alpha(i)<C,
    % b=yi-sum(y.alpha.K(xi,xj). See
    % http://www.mit.edu/~9.520/spring10/Classes/class05-svm.pdf slide 20
    bc_tol=0.000001; % needed to get sensible numbers of SVs. 
    SV_mask = alphas;
    SV_mask(SV_mask<=(0+bc*bc_tol) | SV_mask>=(bc*(1-bc_tol))) =0;
    SV_mask(SV_mask>(0+bc*bc_tol)) =1;
    SV_mask=logical(SV_mask);
    if size(SV_mask,1)==sum(SV_mask) % all vectors are SVs, mask will blow up
        SVs_x=Xtrain;
        SVs_y=ytrain;
    else
        SVs_x=Xtrain(SV_mask,:);
        SVs_y=ytrain(SV_mask,:);
    end
    nSVs=size(SVs_y,1);
    if nSVs>0
        bs=zeros(nSVs,1);
        for svi = 1:1 %size(SVs_y,1) 
            res=0;
            for i = 1:n % support vector terms 
               res=res+ ytrain(i)*alphas(i)*kernel_rbf(SVs_x(svi,:),Xtrain(i,:),kf);
            end
            for i = 1:m % MC terms
               res=res+ betas(i)*(kernel_rbf(MCs(i,:,2),SVs_x(svi,:),kf)-kernel_rbf(MCs(i,:,1),SVs_x(svi,:),kf));
            end
           bs(svi)=SVs_y(svi)-res;
        end
        bias=bs(1,1); %mean(bs);
    else % no SVs
        % use alternate bias calculation, independent of the existence of SVs (see
        % http://fouryears.eu/wp-content/uploads/svm_solutions.pdf pp7-9).
        % Slightly more computationally expensive
        e_LB=-1e9;
        e_UB=1e9;
        for i=1:size(Xtrain,1)
            xi=Xtrain(i,:);
            yi=ytrain(i,1);
            isSV=SV_mask(i);
            res=0;
            for j = 1:n
               res=res+ ytrain(j)*alphas(j)*kernel_rbf(xi,Xtrain(j,:),kf);
            end
            for j = 1:m % MC terms
               res=res+ betas(j)*(kernel_rbf(MCs(j,:,2),xi,kf)-kernel_rbf(MCs(j,:,1),xi,kf));
            end
            ei=yi-res;
            if xor(yi==-1,isSV)
                if ei<e_UB
                    e_UB=ei;
                end
            else
                if ei>e_LB
                    e_LB=ei;
                end
            end
        end
        bias=mean([e_LB,e_UB]); 
        %disp([e_UB bias mean([e_LB,e_UB]) e_LB ]);
    end
    
    % predict ys
%     y_pred=zeros(size(Xtest,1),1);
%     for i=1:size(Xtest,1)
       y_pred=predict_consvm_rbf(Xtest(:,:),alphas,betas,MCs,bias,ytrain,Xtrain,kf); 
%     end
    
end
