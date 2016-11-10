function y = kernel_rbf(X1,X2,sigma)
%kernel_rbf - An implementation of RBF (Gaussian) kernel
%
% INPUTS:
%    X1 - TxP matrix of T X vectors 
%    X2 - TxP matrix of T X vectors 
%
% OUTPUTS:
%    y - Tx1 kernel values K(X1,X2)
%
% Other m-files required: NONE
%

% Author: Chris Bartley
% University of Western Australia, School of Computer Science
% email address: christopher.bartley@research.uwa.edu.au
% Website: http://staffhome.ecm.uwa.edu.au/~19514733/
% Last revision: 30-March-2016

%------------- BEGIN CODE --------------
    Xd=X1-X2;
    l2norms=sum(Xd.*Xd,2);
    y=exp(-l2norms/sigma^2);
end
