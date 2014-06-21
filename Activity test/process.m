clear all;
close all;
clc;
%%% read with sensor data
fid = fopen('homeA-motion\2012-May-25.csv', 'rt');
a = textscan(fid, '%s %f %f', ...
      'Delimiter',',');
fclose(fid);
%%%
data = [a{2} a{3}];
a = a{1};
index = find(strcmpi(a,'master:corner'));
% b = unique(a);
clear a;
data = data(index,:);
date = datevec(data(:,1)/86400 + datenum(1970,1,1));
d = date(:,4)+date(:,5)/60;
figure(1);
plot(d/max(d),data(:,2))
dateaxis('x',15)

%find mind
a = find(date(:,3) == min(date(:,3)));
yesterday_i =a;
min_h = min(date(a,4));
a = find(date(:,4) == min_h);
min_m = min(date(a,5));
%find max
a = find(date(:,3) == max(date(:,3)));
today_i =a;
max_h = max(date(a,4));
a = find(date(:,4) == max_h);
max_m = max(date(a,5));
%calc number of minutes
a = 24*60 - min_h*60 - min_m;
a = a + max_h*60 + max_m;
index = yesterday_i;
minute = min_m;
hour = min_h;
new_d = ones(1,a);
for i=1:a
	minute = minute + 1;
	if (minute >= 60)
		minute = 0;
		hour = hour+1;
	end
	if (hour >= 24)
		hour = 0;
		index = today_i;
	end
% 	hour
% 	minute
    a = find(date(index,4)==hour);
    a = find(date(a,5)==minute);
    disp(strcat(num2str(date(a,4)),':',num2str(date(a,5))));
%     new_d(i) = l_OR(data(a,2));
    if (ismember(1,data(a,2)))
        new_d(i) = 1;
    else
        new_d(i) = 0;
    end
end

figure(2);
a=1:length(new_d);
a=a/length(a);
a = a';
plot(a,new_d);
dateaxis('x',15)