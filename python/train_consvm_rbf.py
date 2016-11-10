import numpy as np
from kernel_rbf import kernel_rbf
from predict_consvm_rbf import predict_consvm_rbf
from quadprog_cvx import quadprog

s_FREE=0
s_UPPER_BOUND=1
s_LOWER_BOUND=2
def train_consvm_rbf(ytrain,Xtrain,bc,kf,Xtest, MCs):
#train_consvm_rbf - trains a C-SVM binary classifier with monotone constraints.
#
# DESCRIPTION:
# In essence the resulting SVM classifier will respect f(x')>=f(x) for all
# pairs of constraint points (x,x') in constraint set MCs.
# Core algorithm is as described in:
# Chen, C.C. & Li, S.T., 2014. Credit rating with a monotonicity-constrained 
# support vector machine model. Expert Systems with Applications, 41(16), pp.7235�7247.
#
# INPUTS:
#    ytrain - Nx1 vector of binary class y (-1 / +1)
#    Xtrain - NxP matrix of P-dimensional X data. 
#    bc - Box constraint for C-SVM
#    kf - Kernel Factor for RBF kernel.
#    Xtest - TxP test dataset of length T (may use Xtrain if no separate
#    test partition)
#    MCs - MxPx2 matrix of M constraints where f(x_m')>=f(x_m) is guaranteed for 
#    x_m=MCs(m,:,1) and x_m'=MCs(m,:,2). You can create these constraintsets for
#    PM-SVM using gen_constrset_pmsvm_nc or gen_constrset_pmsvm_adaptive.
#
# OUTPUTS:
#    alphas - Nx1 vector of solution for alpha Lagrangian multipliers (non-zero values correspond to support vectors)
#    betas - Mx1 vector of solution for beta Lagrangian multipliers (non-zero values correspond to support constraints)
#    y_pred - Tx1 vector of predicted y values for Xtest
#    bias - SVM bias scalar
#    H - Solution matrix solved by quadprog
#    minEig - minimum eigenvalue of solution matrix - if minEig<0, and diagonal
#    correction of +2*abs(minEig) is included in H to guarantee problem
#    convexity (Tikhonov regularisation).
#
# EXAMPLE:
#   To create 500 constraints, with increasing features [2 5] and
#   decreasing features [3 7], with no mid pts but with greedy mms and duplicate 
#   detection, and then solve the PM-SVM model:
#   MCs=gen_constrset_pmsvm_nc(Xtrain, 500,[2 5],[3 7],{'5','0','1','1'}) 
#   [alphas,betas, y_pred,bias,H, minEig] = train_consvm_rbf(ytrain,Xtrain,bc,kf,Xtest, MCs)
#
# Other m-files required: kernel_rbf
#
# Author: Chris Bartley
# University of Western Australia, School of Computer Science
# email address: christopher.bartley@research.uwa.edu.au
# Website: http://staffhome.ecm.uwa.edu.au/ not 19514733/
# Last revision: 10-Nov-2016
    # prelimsminE
    n=Xtrain.shape[0]
    if MCs==[] or MCs is None:
        m=0
    else:
        m=MCs.shape[0]
    # construct solution matrix
    G11=np.zeros([n,n],dtype='float64')
    for row in np.arange(0,n):
        for col in np.arange(0,n):
            #Xd=Xtrain(row,:)-Xtrain(col,:)
            G11[row,col]=ytrain[row]*ytrain[col]*kernel_rbf(Xtrain[row,:],Xtrain[col,:],kf) #(-Xd*Xd'/kf^2)
    G12=np.zeros([n,m],dtype='float64')
    for i in np.arange(0,n):
        for j in np.arange(0,m):
            G12[i,j]=ytrain[i]*(kernel_rbf(MCs[j,:,1],Xtrain[i,:],kf)-kernel_rbf(MCs[j,:,0],Xtrain[i,:],kf)) #(-Xd*Xd'/kf^2)
    G22=np.zeros([m,m],dtype='float64')
    for i in np.arange(0,m):#
        for j in np.arange(0,m):
            G22[i,j]=kernel_rbf(MCs[i,:,1],MCs[j,:,1],kf)  +  kernel_rbf(MCs[i,:,0],MCs[j,:,0],kf) - kernel_rbf(MCs[i,:,1],MCs[j,:,0],kf) - kernel_rbf(MCs[i,:,0],MCs[j,:,1],kf) #(-Xd*Xd'/kf^2)
            
    # lazy way to correct small asymmetries in G22 (1e-22 differences!)
    # Otherwise quadprog gives warning
    #G22=(G22+G22.T)/2
    # create quadprog variables
    G=np.vstack((np.hstack((G11,G12)),np.hstack( (G12.T, G22))))
    # clean G of super low values, which can cause solve_qp to fall over (giving 'nan' values)
    G=clean_qp_matrix(G)
    f=-1*np.ones((n+m,1))
    f[n:(n+m)]=0.
    A=None
    b=None
    Aeq=(np.vstack([ytrain.reshape([n,1]), np.zeros([m,1])])).T
    beq=0.0 #np.zeros(1,1)
    lb=np.zeros([n+m,1])
    ub=bc*np.ones([n+m,1])
    ub[n:n+m]=1e10 #bc*100# effectively remove upper bound on betas
    # solve for weights
    # Use Tikhonov regularisation to ensure positive definite H
    minEig=get_min_eig(G)
    H=tikhonov_regularise_posdef(G)
    
    # Build QP matrices
    #Minimize     1/2 x^T H x - f^T x
    #Subject to   A.T x >= b  
    alphasbetas=quadprog(H,f,A,b,Aeq,beq,lb,ub) 
    alphasbetas=np.ravel(alphasbetas)
    alphas=alphasbetas[0:n]
    betas=alphasbetas[n:n+m]
    # Calculate bias 'bias'. For support vectors 0<alpha(i)<C,
    # CALCULATE BIAS (rho in libsvm) - original technique
    # b=yi-sum(y.alpha.K(xi,xj). See
    # http://www.mit.edu/ not 9.520/spring10/Classes/class05-svm.pdf slide 20
    bc_tol=1e-2*np.min([10000,bc])#np.min([0.99*np.max(alphas),1e-2*np.min([10000,bc])]) # needed to get sensible bias calculation. maximum of 1000 on box constraint because very high box constraints can cause this tolerance to be too great 
    # identify support vectors ABSOLUTE TOLERANCE
    SV_mask = alphas.copy()
    SV_mask[SV_mask<(0+bc_tol) ] =0
    SV_mask[SV_mask>=(0+bc_tol)] =1
    SV_mask=SV_mask.astype('bool')
    SV_mask_on_margin = alphas.copy()    
    filter_s=np.logical_or(SV_mask_on_margin<=(0+bc_tol) ,  SV_mask_on_margin>=(bc-bc_tol))
    SV_mask_on_margin[filter_s] =0
    SV_mask_on_margin[SV_mask_on_margin>=(0+bc_tol)] =1
    SV_mask_on_margin=SV_mask_on_margin.astype('bool')
    
    if len(SV_mask)==np.sum(SV_mask): # all vectors are SVs, mask will blow up
        SVs_x=Xtrain
        SVs_y=ytrain
    else:
        SVs_x=Xtrain[SV_mask,:]
        SVs_y=ytrain[SV_mask]
    if len(SV_mask_on_margin)==np.sum(SV_mask_on_margin): # all vectors are SVs, mask will blow up
        SVs_on_margin_x=Xtrain
        SVs_on_margin_y=ytrain
    else:
        SVs_on_margin_x=Xtrain[SV_mask_on_margin,:]
        SVs_on_margin_y=ytrain[SV_mask_on_margin]
    nSVs=SVs_y.shape[0]
    nSVs_on_margin=np.sum(SV_mask_on_margin)
    if nSVs_on_margin>0: #&& False # have at least one SV on the margin
        if len(SVs_on_margin_y) ==1:
            res=np.asarray(predict_consvm_rbf(SVs_on_margin_x,alphas,betas,MCs,0,ytrain,Xtrain,kf)[1][1])
        else: # more than onee
            res=np.asarray(predict_consvm_rbf(SVs_on_margin_x,alphas,betas,MCs,0,ytrain,Xtrain,kf)[1][:,1])
        bs=np.ravel(SVs_on_margin_y)-res
        bias=np.median(bs) #bs(1,1) #mean(bs)
    else: # no SVs on margin - use non-margin SVs via
        # alternate bias calculation, independent of the existence of SVs on margin (see
        # http://fouryears.eu/wp-content/uploads/svm_solutions.pdf pp7-9).
        # Slightly more computationally expensive
        e_LB=-1e9
        e_UB=1e9
        for i in np.arange(0,Xtrain.shape[0]):
            xi=Xtrain[i,:]
            yi=ytrain[i]
            isSV=SV_mask[i]
            res=predict_consvm_rbf(xi,alphas,betas,MCs,0,ytrain,Xtrain,kf)[1][1]
            ei=yi-res
            if (yi==-1)^(not isSV): # XOR
                if ei<e_UB:
                    e_UB=ei
            elif (yi==-1 )^ ( isSV): #XOR
                if ei>e_LB:
                    e_LB=ei
        if e_LB==-1e9:
            bias=e_UB
        elif  e_UB==1e9:
            bias=e_LB
        else:
            bias=np.mean([e_LB,e_UB])

    # predict ys
    y_pred=predict_consvm_rbf(Xtest,alphas,betas,MCs,bias,ytrain,Xtrain,kf) 
    y_pred=y_pred[0]
#     end
    return [alphas,betas, y_pred,bias,H, minEig]#np.min(minEig1,minEig2)] 
  

def is_pos_def(G):
    try:
        #x=cholesky(G)# fastest way to check for pd
        #Htest=H.astype(np.single)
        minEig=get_min_eig(G)
        return minEig>0 #and minEig2>0 and minEig3>0
    except: #cholesky error, LinAlgError assumed, not pos def
        return False

def get_min_eig(G):
    # This is a horrid piece of code. It seems pythons linalg.eig, linalg.eigh, and linalg.eigvals
    # often give different results for the smallest eigenvalues,  and I need to check all of them, 
    # otherwise quadprog solve_qp sometimes fails on non-pd matrix. 
    # It also gives different results when cast as a single, and I need to 
    # check as double and single!!! Yuck!!!
    minE=1e9
    for typ in [np.single,np.double]:
        H=G.astype(typ)#np.single)
#        [eigenvaluesH,eigvectors]=np.linalg.eigh(H)
#        minEig1=np.min(eigenvaluesH.real)
#        [eigenvaluesH,eigvectors]=np.linalg.eig(H)
#        minEig2=np.min(eigenvaluesH.real)
        eigenvaluesH=np.linalg.eigvals(H)
        minEig3=np.min(eigenvaluesH.real)
        minE=np.min([minE,minEig3])#minEig1,minEig2,minEig3])
    return minE #np.min([minEig1,minEig2,minEig3])
    
def clean_qp_matrix(G,tol=1e-25):
    H=G.copy() #((G+G.T)/2.).copy()
    for i in np.arange(0,H.shape[0]):
        for j in np.arange(0,H.shape[1]):
            if abs(H[i,j])<tol:
                H[i,j]=0.
    return H
    
def tikhonov_regularise_posdef(G):
    #This approach ramps up the diagonal 
    # increment until both report satisfactory minimum eigen values.
    H=G.copy()
    minEig=get_min_eig(H)
    if not is_pos_def(H): #or minEig2<0:
        posdef=False
        increxp=1
        increment=np.abs(minEig)
        while not posdef:
            increment=increment*2#**increxp*np.abs(minEig)#np.min(minEig1,minEig2))
            H=G.copy()+increment*np.eye(G.shape[0]) 
#            Htest=H.astype(np.single)
#            [eigenvaluesH,eigvectors]=np.linalg.eigh(Htest)
#            minEig1=np.min(eigenvaluesH)
#            [eigenvaluesH,eigvectors]=np.linalg.eig(Htest)
#            minEig2=np.min(eigenvaluesH)
#            eigenvaluesH=np.linalg.eigvals(Htest)
#            minEig1=np.min(eigenvaluesH)
            if not is_pos_def(H): #or minEig2<0:
                increxp=increxp+1
            else:
                posdef=True
    return H