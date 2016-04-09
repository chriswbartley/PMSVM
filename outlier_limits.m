% calculates the upper and lower limits for outliers from a vector of
% values by (a) applying the best normalising transformation; and (b) using
% mean +/- 3 standard deviations.
% if x has more than 1 column, the results will too.
function [low_lim,upp_lim]=outlier_limits(X,stdevs)
    switch nargin
        case 2
            % all set
        case 1
            stdevs=3;
    end

    %check if need to transpose X
    if size(X,1)>1
         X_copy=X; 
    else
       X_copy=X'; 
    end
    %10: get optimum lambdas 
    constants=zeros(1,size(X_copy,2));
    lambdas=zeros(1,size(X_copy,2));
    low_lim=zeros(1,size(X_copy,2));
    upp_lim=zeros(1,size(X_copy,2));
    for col =1:size(X_copy,2)
        constants(1,col)=iif(min(X_copy(:,col))<=0,-min(X_copy(:,col))+1e-5,0);
        [x_trans,lambda] = boxcox(X_copy(:,col)+constants(1,col));
        lambdas(1,col)=lambda;
        mean_trans=mean(x_trans);
        std_trans=std(x_trans);
        llim=mean_trans-stdevs*std_trans;
        ulim=mean_trans+stdevs*std_trans;
        if lambda>=0 % need to check for impossible values
            llim=iif(llim<=0,min(x_trans)*0.5,llim);
        else
            ulim=iif(ulim>=0,max(x_trans)*0.5,ulim);
        end
        
        low_lim(1,col)=bcTransformInv(lambda,min([llim ulim]))-constants(1,col);
        upp_lim(1,col)=bcTransformInv(lambda,max([llim ulim]))-constants(1,col);
    end
end