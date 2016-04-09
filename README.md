# PMSVM
Partially Monotone SVM

This repository provides all MATLAB code necessary to use build Partially Monotone SVM models for binary classification problems. The framework is described in Bartey, Liu and Reynolds 'Effective Knowledge Integration in SVM for Improved Accuracy' 2016 (submitted to ECML PKDD). Please see the paper for further details.

A typical process to build a partially monotone knowledge integrated model is:

1. Use grid search with k-fold cross-validation to find the optimal hyperparameters (box constraint and RBF kernel factor) for the UNCONSTRAINED SVM (use train_consvm_rbf() with MCs=[], or MATLAB's fitcsvm for RBF kernel (which uses SMO instead of quadprog and is faster)).

2. Solve the optimal unconstrained SVM model using train_consvm_rbf() and the optimal hyperparameters, to obtain the alphas and bias values needed to finalise the model.

3. Decide which features are suggested monotone in the response, based on domain knowledge.

4. Check that these features are sensible using calc_mcc_interp_pmsvm_rbf() to calculate the proportion of data points for which these features are monotone increasing, monotone decreasing, monotone increasing AND decreasing, or no change. Use Equation 10 to calculate the disagreement metric (dis=(Ndec+0.5Nincdec)/(Ninc+Ndec+Nincdec)), and if dis>90% do NOT constraint that feature.

5. Check unconstrained SVM model monotonicity in suggested features using calc_mcc_pmsvm_rbf() to measure monotonicity levels for each feature (MCC). If they are already almost 100%, adding constraints will not change the model much.

6. Build a constraint set using gen_constrset_pmsvm_adaptive() or gen_constrset_pmsvm_nc(). If you are unsure what to use, gen_constrset_pmsvm_adaptive() is suggested with BasePtsUnbounded='ub_off' and a large number of constraints. This will create one constraint per non-monotone point and is generally sufficient. To use it with it's default formulation:MCs=MCs=gen_constrset_pmsvm_adaptive(Xtrain, 500000,incrfeats,decrfeats 7],{'randseln','NMTpt_prop','randseln','baseanchor_none','endanchor_none','base_noprior','od_off','ub_off','dupchk_on',kf,ytrain,Xtrain,SVM_alphas,SVM_bias})

7. Solve the CONSTRAINED SVM using train_consvm_rbf(). 

8. You can then assess the resulting model with predict_consvm_rbf() for predications, and calc_mcc_pmsvm_rbf() to measure final monotonicity levels for each constrained feature (MCC).
