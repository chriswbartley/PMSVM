from predict_consvm_rbf import predict_consvm_rbf
class model_consvm_rbf:
    def __init__(self,alphas,betas,MCs,bias,y_train,X_train,kf,bc=0):
        self.alphas=alphas
        self.betas=betas
        self.MCs=MCs
        self.bias=bias
        self.y_train=y_train
        self.X_train=X_train
        self.kf=kf
        self.bc=bc
    def predict(self,Xs):
        p= predict_consvm_rbf(Xs,self.alphas,self.betas,self.MCs,self.bias,self.y_train,self.X_train,self.kf)
        return p[0]

