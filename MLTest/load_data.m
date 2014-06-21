function [data] = load_data(month)

data = [];
for i=1:31
    str = ['data/2012-' month '-' int2str(i) '.csv'];
    if exist(str, 'file') == 2
        a = csvread(str);
        data = [data; a];
    else
        disp([str 'does not exist - skipping']);
    end
end

end