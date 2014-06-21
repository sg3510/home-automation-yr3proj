
for index_a=1:10
%% import env  data
% clear all;
% clc;
% close all;
fid = fopen('p_zdata/env.csv', 'rt');
a = textscan(fid, '%s %s %f %f', ...
      'Delimiter',',');
fclose(fid);
%prepare and format it
c = cell2mat(a{1});
date_in = datevec(c(:,:),'dd/mm/yyyy');
c = cell2mat(a{2});
t = datevec(c(:,:),'HH:MM:SS');
time_in = t(:,4:6);
%get temp data
temp = a{3};
light_in = a{4};
%% import weather data
fid = fopen('p_zdata/weather_dat.csv', 'rt');
a = textscan(fid, '%s %s %f %f %f %f %f %f', ...
      'Delimiter',',');
fclose(fid);
%prepare and format it
c = cell2mat(a{1});
date_weather = datevec(c(:,:),'dd/mm/yyyy');
c = cell2mat(a{2});
time_weather = datevec(c(:,:),'HH:MM:SS');
time_weather = time_weather(:,4:6);
%get outsidetemp data
out_temp = a{3};
pressure = a{4};
humidity = a{5};
wind_speed = a{6};
w_cond = a{8};
%% bit of post processing
sensor = csvread('p_zdata/sensor_env.csv');
%find intersect
% intersect = ismember(date_in,date_weather,'rows');
% index = find(intersect==1);
% date_in = date_in(index);
% time_in = time_in(index);
% temp = temp(index);
% sensor = sensor(index);
% index_a = 1;
time = time_in(:,1)+time_in(:,2)/60;
index = find(sensor==index_a);
%% combine data
data = zeros(length(date_weather),11);
% only get for one sensor
date_in = date_in(index,:);
time_in = time_in(index,:);
temp = temp(index);
index = 0;
a=length(date_weather)*length(date_in);
prev = -1;
for i=1:length(date_weather)
    for j=1:length(date_in)
        if ((date_weather(i,2)==date_in(j,2))&&(date_weather(i,3)==date_in(j,3))&&(date_weather(i,1)==date_in(j,1)))
             b=100*(i+i*j)/a;
            if (round(b) == prev)
            else
                prev = round(b);
                disp([num2str(b) '% for index=' num2str(index_a)]);
            end
            if(time_in(j,1)==time_weather(i,1))
%                 disp([num2str(date_weather(i,1)) '-' num2str(date_weather(i,2)) '-' num2str(date_weather(i,3))])
%                 disp([num2str(time_in(j,1)) ':' num2str(time_in(j,2))])
                index = index + 1;
                data(index,1) = date_weather(i,1);
                data(index,2) = date_weather(i,2);
                data(index,3) = date_weather(i,3);
                data(index,4) = time_weather(i,1);
                data(index,5) = time_in(j,2);
                data(index,6) = temp(j);
                data(index,7) = out_temp(i);
                data(index,8) = pressure(i);
                data(index,9) = humidity(i);
                data(index,10) = wind_speed(i);
                data(index,11) = w_cond(i);
            end
        end
    end
end
save(['data_zp_sensor' num2str(index_a)],'data')
end
%% plot
figure(1);
clf;
hold on;
windowSize = 100;
a = filtfilt(ones(1,windowSize)/windowSize,1,out_temp(:));
scatter(date_in(index,2)+date_in(index,3)/30,temp(index),'.');
scatter(date_weather(:,2)+date_weather(:,3)/30,a,'.');
hold off;