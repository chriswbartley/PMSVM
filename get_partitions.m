function [nparts,parts]=get_partitions(vals,maxPartitions,isIncreasing,boolSmartCategoryIdent)
%get_partitions - determines a set of partitions for a vector of data
%values
% DESCRIPTION:
%   From a given set of feature values, it finds a uniform set of values
%   that partition the range.
%
% INPUTS:
%    vals - Nx1 vector of values
%    maxPartitions - Maximum number of partitions to return. If number of
%       unique values is less than this, the (number of unique values -1) will be
%       used. If data is scalar (continuous) this is the number of partitions
%       that will be returned.
%    isIncreasing - 1 to return partition values increasing, otherwise 0 they
%       will be decreasing.
%    boolSmartCategoryIdent - 1 to attempt to automatically identify
%       whether this is categorical (rather than scalar) data, otherwise 0. For 
%       smart identification, if the number of unique values is less than 15, 
%       it will be treated as categorical.
%
% OUTPUTS:
%    nparts - Number of partitions found
%    parts - PPx1 vector of partitions found.
%
% Other m-files required: none

% Author: Chris Bartley
% University of Western Australia, School of Computer Science
% email address: christopher.bartley@research.uwa.edu.au
% Website: http://staffhome.ecm.uwa.edu.au/~19514733/
% Last revision: 30-March-2016

%------------- BEGIN CODE --------------
    % determine  unique values (sorted lowest to highest)
    uniq=unique(vals);
    n_uniq=numel(uniq);
    % if number of unique values is less than maxPartitions, use unique
    % values
    if n_uniq==1
        nparts=0;
        parts=uniq;
    else    
        if boolSmartCategoryIdent
            if n_uniq<=(maxPartitions+1)
                parts=uniq';
                nparts=n_uniq-1;
            else
                if n_uniq<15 % assume categorical, use values from uniq
                    approxincr=(n_uniq-1)/maxPartitions;
                    parts=zeros(1,maxPartitions+1);
                    for i=0:maxPartitions
                        parts(1,i+1)=uniq(round(1+approxincr*i));
                    end
                    nparts=maxPartitions;
                else % assume continuous, divide (min/max) into maxParts
                    minu=min(uniq);
                    maxu=max(uniq);
                    incr=(maxu-minu)/maxPartitions;
                    parts=minu:incr:maxu;
                    nparts=maxPartitions;
                end
            end
        else % use dumb partitions, do not attempt to identify categories, treat all as continuous
            minu=min(uniq);
            maxu=max(uniq);
            incr=(maxu-minu)/maxPartitions;
            parts=minu:incr:maxu;
            nparts=maxPartitions;
        end
        % if not increasing, revierse partitions
        if ~isIncreasing
            parts=sort(parts,'descend');
        end
        if size(parts,1)>size(parts,2)
           parts=parts'; 
        end
    end
end
