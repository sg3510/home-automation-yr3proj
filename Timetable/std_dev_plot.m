fid = fopen('text.txt');

tline = fgets(fid);
a = zeros(5,1);
while ischar(tline)
    a = [a,sscanf(tline,'Acc Avg:%f%% Std.Dev.:%d%% w/day len=%d s_p=%d s_f=%d')];
    tline = fgets(fid);
end

fclose(fid);