clear all;
clc;
data = load_env_data('data');
data = load_data('Jun');
%convert to celsius
data(:,2:3) = (data(:,2:3) -32)*5/9;
% data = [data(:,1:9) data(:,11:3)];
currentDateAsVector = datevec(datenum(1970,1,1)+data(:,1)/86400);
X = data(:,3:end);
y = data(:,2);
m = length(y);
n = length(X(2,:));
if exist('timeday.mat', 'file') == 2
    load timeday.mat
    disp('loaded');
else
    time = currentDateAsVector(:,4)+currentDateAsVector(:,5)/60;
    day = [strcat(num2str(currentDateAsVector(:,2)),'-') strcat(num2str(currentDateAsVector(:,3)),'-') num2str(currentDateAsVector(:,1))];
    day = weekday(day);
    save('timeday','day','time')
end
%% add data to X
X = [X time day];
%recalc size
m = length(y);
n = length(X(2,:));
%%
map = day/7*ones(1,3);
figure;
sun = find(day==1);
mon = find(day==2);
tues = find(day==3);
wed = find(day==4);
thur = find(day==5);
fri = find(day==6);
sat = find(day==7);
map(sun,:) = ones(length( map(sun)),1)*[0 1 0];
map(mon,:) = ones(length( map(mon)),1)*[0 1 1];
map(sat,:) = ones(length( map(sat)),1)*[0 1 1];
map(tues,:) = ones(length( map(tues)),1)*[1 0 1];
hold on;
scatter3(day,time,data(:,2),4,map);
colormap(map)
hold off;
xlabel('day of the week')
ylabel('time')
zlabel('temp')
%% find theta values
[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];
% Choose some alpha value
alpha = .1;
num_iters = 100;

% Init Theta and Run Gradient Descent 
theta = zeros(n+1, 1);
[theta, J_history, theta_h] = gradientDescentMulti(X, y, theta, alpha, num_iters);
figure(1);
plot(J_history);
figure(2);
plot(theta_h);

theta_n = normalEqn(X,y);

%% theta for time
X = [time time.^2 time.^3 time.^4 day day.^2 day.^3 day.^4 data(:,3)];
m = length(y);
n = length(X(2,:));
% [X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];
clc;
theta_n = normalEqn(X,y);

d = 1:7;
t = 0:0.1:24;
a = theta_n(1) + theta_n(2)*t + theta_n(3)*t.^2 + theta_n(4)*t.^3 + theta_n(5)*t.^4;
b = theta_n(1) + theta_n(6)*d + theta_n(7)*d.^2 + theta_n(8)*d.^3 + theta_n(9)*d.^4;
t_l = length(t);
d=0;
for d = 1:7
    for j = 1:t_l;
        y_o(j,d) = theta_n(1) + theta_n(2)*t(j) + theta_n(3)*t(j).^2 + theta_n(4)*t(j).^3 + theta_n(5)*t(j).^4 + theta_n(6)*d + theta_n(7)*d.^2 + theta_n(8)*d.^3 + theta_n(9)*d.^4;
    end
end
surf(1:7,t/24,y_o)
xlabel('Day');
xlim([1 7])
ylabel('Time');
dateaxis('y', 16)
% set day
days = ['Sun';
          'Mon';
          'Tue';
          'Wed';
          'Thu';
          'Fri';
          'Sat'];
set(gca,'XTickLabels',days)
set(gca,'Ydir','reverse','Xdir','reverse')
zlabel('Temperature');