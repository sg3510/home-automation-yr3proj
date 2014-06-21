clear all;
clc;
load('data_zp_sensor7.mat');
%remove duplicates
[a ia ~] = unique(data(:,1:5),'rows');
data = data(ia,:);
clear a ia;
%get month data
month = ones(1,9);
for i=1:max(data(:,2))
    index = find(data(:,2)==i);
    month(i) = max(data(index,3));
end
a = find(isnan(data(:,8)));
data(a,8) = 100;
day = data(:,2) + data(:,3)./month(data(:,2))';
day_week = weekday(datenum(data(:,1:3)));
clear i a index month;
%get time
time = data(:,4)+data(:,5)/60;

%% start
% X = month day time temperature,pressure,humidity,wind_speed, weather
y = data(:,6);
X = [data(:,2) day_week time data(:,7:11) data(:,11).^2];
[m n] = size(X);
[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];
theta_n = normalEqn(X,y);

%% 3D plot
y = data(:,6);
X = [time time.^2 time.^3 time.^4 data(:,7) data(:,7).^2 data(:,7).^3 data(:,7).^4 data(:,7).^5  data(:,7).^6];
X = [ones(m, 1) X];
theta_n = normalEqn(X,y);

t = 0:0.1:24;
t_l = length(t);

for d = 1:length(0:0.1:30)
    for j = 1:t_l;
        y_o(j,d) = theta_n(1) + theta_n(2)*t(j) + theta_n(3)*t(j).^2 + theta_n(4)*t(j).^3 + theta_n(5)*t(j).^4 + theta_n(6)*(d/10) + theta_n(7)*(d/10).^2 + theta_n(8)*(d/10).^3 + theta_n(9)*(d/10).^4 + theta_n(10)*(d/10).^5 + theta_n(11)*(d/10).^6;
    end
end
surf(0:0.1:30,(0:0.1:24)/24,y_o)