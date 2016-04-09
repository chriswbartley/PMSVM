# PMSVM
Partially Monotone SVM

This repository provides all MATLAB code necessary to use build Partially Monotone SVM models for binary classification problems. The framework is described in Bartey, Liu and Reynolds 'Effective Knowledge Integration in SVM for Improved Accuracy' 2016 (submitted to ECML PKDD). Please see the paper for further details.

The basic process for using knowledge about likely features for which the resonse is partially monotone, the basic process is:
1. Use grid search to find optimal box constraint and RBF kernel factor values for the UNCONSTRAINED SVM (use train_consvm_rbf() with MCs=[]).
2. Decide which features are suggested monotone in the response.
3. Check that these features are sensible using calc_mcc_interp_pmsvm_rbf() to calculate the proportion of data points for which these features are monotone increasing, monotone decreasing, monotone increasing AND decreasing, or no change. Use Equation 10 to calculate the disagreement metric, and if dis>90% do NOT constraint that feature.
4. Build a constraint set using gen_constrset_pmsvm_adaptive() or gen_constrset_pmsvm_nc(). If you are unsure what to use, gen_constrset_pmsvm_adaptive() is suggested with BasePtsUnbounded='ub_off' and a large number of constraints will fully automatically create the constraints (one per non-monotone point). To use it with it's default formulation:
