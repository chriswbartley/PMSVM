import numpy as np
from kernel_rbf import kernel_rbf
def  predict_consvm_rbf(predXs,alphas,betas,MCs,b,y,X,kf):
#predict_consvm_rbf - predict a pm-svm model for set of inputs predXs
# DESCRIPTION:
# Predicts a constrained SVM using algorithm is as described in:
# Chen, C.C. & Li, S.T., 2014. Credit rating with a monotonicity-constrained 
# support vector machine model. Expert Systems with Applications, 41(16), pp.72357247.
#
# INPUTS:
#    predXs - TxP matrix of X vectors where class is to be predicted
#    alphas - Nx1 vector of solution for alpha Lagrangian multipliers (non-zero values correspond to support vectors)
#    betas - Mx1 vector of solution for beta Lagrangian multipliers (non-zero values correspond to support constraints)
#    MCs - MxPx2 matrix of M constraints where f(x_m')>=f(x_m) is guaranteed for 
#           x_m=MCs(m,:,1) and x_m'=MCs(m,:,2)
#    b - Bias term
#    y - Original Nx1 y vector used to train constained SVM
#    X - Original NxP X vector used by trained constained SVM
#    kf - RBF kernel factor
#
# OUTPUTS:
#    classes - Tx1 predicted classes based on predXs
#    scores - Tx1 predicted real value of constrained SVM (prior to converting to sign)
#
# Other m-files required: kernel_rbf
#
# Author: Chris Bartley
# University of Western Australia, School of Computer Science
# email address: christopher.bartley@research.uwa.edu.au
# Website: http://staffhome.ecm.uwa.edu.au/ not 19514733/
# Last revision: 30-March-2016
#------------- BEGIN CODE --------------
    n = y.shape[0]
    if len(predXs.shape)<2:
        predXs=predXs.reshape((1,predXs.shape[0]))
    if MCs==[] or MCs is None or MCs.shape[0]==0:
        m=0
    else:
        m=MCs.shape[0]
    numPredXs=predXs.shape[0]
    ys=np.zeros([numPredXs,1])
    scores=np.zeros([numPredXs,2])
#     # pre-filter on support vectos
#     tic
    sv_indexes=np.abs(alphas)>1e-8
    alphas=alphas[sv_indexes]
    y=y[sv_indexes]
    X=X[sv_indexes,:]
    # pre-filter betas
    if betas==[] or betas is None:
        hasbetas=False
    else:
        if len(betas[betas>0]) !=0:
            sc_indexes=np.abs(betas)>1e-8
            betas=betas[sc_indexes]
            MCs_filtered=np.zeros([len(betas),MCs.shape[1],2])
            MCs_filtered[:,:,0]=MCs[sc_indexes,:,0]
            MCs_filtered[:,:,1]=MCs[sc_indexes,:,1]
        hasbetas=len(betas[betas>0])>0
    # calculate results for each prediction point
    for ix in np.arange(0,numPredXs):
        res=np.sum(np.multiply(np.multiply(alphas,y),kernel_rbf(X,np.tile(predXs[ix,:],(X.shape[0],1)),kf)))
        if hasbetas:
            res=res+ np.sum(np.multiply(betas,(kernel_rbf(MCs_filtered[:,:,1],np.tile(predXs[ix,:],(len(betas),1)),kf)-kernel_rbf(MCs_filtered[:,:,0],np.tile(predXs[ix,:],(len(betas),1)),kf))))
        ys[ix]=1 if (res+b)>0 else -1
        scores[ix,:]=[-1*(res+b), (res+b)]

    classes=ys[0] if numPredXs==1 else ys
    scores=scores[0,:] if numPredXs==1 else scores
    return [classes,scores]