fid=fopen('c_3data.txt','wt');
for i=1:length(house_state)
    fprintf(fid,[num2str(house_state(i)) ',' datestr(t_range(i),'dd-mmm-yyyy HH:MM:SS') '\n']);
end
fclose(fid);