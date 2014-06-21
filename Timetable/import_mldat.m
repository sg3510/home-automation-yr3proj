clc; clear all;
[num, txt, all] = xlsread('Timetable-1_data.csv');
dat = 'dd/mm/yyyy HH:MM:SS';
a = ones(1,2688);
for i=1:2688
    try
        a(i) = datenum(all{i},dat);
    catch
        a(i) = datenum(all{i},'dd/mm/yyyy');
    end
end
dat = a;
num = num(:,2);

%% plot autocorrelation
corr_num = (num -mean(num))/mean(num);
corr_num = xcorr(corr_num,'bias');
corr_num = corr_num(2688:end);
plot(corr_num)

%% plot a day
samples = 96; %defines number of samples in a day
day = 28;
plot(num((day-1)*samples+1:day*samples))
axis tight