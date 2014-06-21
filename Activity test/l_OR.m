function [out] = l_OR(in_dat)
%L_OR Summary of this function goes here
%   Detailed explanation goes here
out = 0;
for i=1:length(in_dat)
    out = out|in_dat(i);
end

end

