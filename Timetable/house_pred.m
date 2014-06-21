clc;
clear;
clear all;
close all;
% load housestate.mat
load usualday.mat
t_range = dat;
house_state = num';
sample = 288; %number of samples in a day
ratio = 3;
sample = sample/ratio;
clear dat;
clear num;
% for day=1:1
%     figure;
%     plot(house_state((day-1)*sample+1:day*sample));
%     axis tight;
% end
day = round(60*rand);
%% predict based on day
day = 7; %day datenum -> (day-1)*sample+1
pred = [1 6 12 24 36 48 60 72 96 144 184 288]/ratio;
pred = ceil(pred);
len = 5;
acc_3d = [1 1 1];
for len = 5:20 %how many days to average from
    for pred_d=pred %how many hours to check for accuracy
        for day=2:length(house_state)/sample-2 %loop through days
            time_o = 180/ratio ;%equivalent to 7:30 AM
            day_t_index = time_o+(day-1)*sample+1;
            % disp(datestr(t_range(day_t_index)))
            %store data for future (i.e. to be predicted data)
            day_f = house_state(day_t_index:day_t_index+sample-1);
            %store data of past day
            day_p = house_state(day_t_index-sample:day_t_index-1);
            states = house_state;
            %make data 3 - effecticely makign it the greatest hamming distance from all
            states(day_t_index-sample:day_t_index+sample-1) = 3*ones(sample*2,1);
            for i=1:length(house_state)/sample-2
                %(i-1)*sample+time_o+1
                %i*sample+time_o
            %     disp(datestr(t_range((i-1)*sample+time_o+1)))
            %     disp(datestr(t_range(i*sample+time_o)))
            % figure;
            % subplot(2,1,1), plot(states((i-1)*sample+time_o+1:i*sample+time_o))
            % subplot(2,1,2), plot(day_p)
                a = [day_p;states((i-1)*sample+time_o+1:i*sample+time_o)];
                D(i) = pdist(a,'hamming');
            end
            [sortedValues,sortIndex] = sort(D,'ascend');
            maxIndex = sortIndex(1:len); %get minimum hamming distance days
            %make sure not to go out of range
            if any(maxIndex == 59)
                index = find(maxIndex==59);
                maxIndex(index) = sortIndex(len+1);
            end
            days_hd = ones(sample,1)';
            % future
            for i=maxIndex
                days_hd = [days_hd ;house_state(i*sample+time_o+1:(i+1)*sample+time_o)];
            end
            days_hd = days_hd(2:end,:)';
            days_hd = mode(days_hd');
            % figure;
            % subplot(2,1,1), plot(days_hd)
            % subplot(2,1,2), plot(day_f)
            days_hd(days_hd==2) = 1;
            day_f(day_f==2) = 1;
            acc(day) = 100*(1-pdist([days_hd(1:pred_d); day_f(1:pred_d)],'hamming'));
            % disp(sprintf('Accuracy:%2.1f%%',acc(day)));
        end
        acc = acc(2:end);
        min = pred_d*5*ratio;
        hour = 0;
        while (min >= 60)
            hour = hour +1;
            min = min - 60;
        end
        disp(sprintf('Accuracy for %d Hour(s) and %d minutes into the future with %d days regression:%2.1f%%',hour,min,len,mean(acc)));
        acc_3d = [acc_3d;pred_d*5,len,mean(acc)];
    end
end
acc_3d = acc_3d(2:end,:);
% current day
%{
for i=maxIndex
    days_hd = [days_hd ;states((i-1)*sample+time_o+1:i*sample+time_o)];
end
days_hd = days_hd(2:6,:)';
figure;
subplot(2,1,1), plot(mode(days_hd'))
subplot(2,1,2), plot(day_p)
%}
%% test data
%{
a = house_state((day-1)*sample+1:day*sample);
for day=1:length(house_state)/sample
    D(day) = pdist([a;house_state((day-1)*sample+1:day*sample)],'hamming');
end
% gets vlaues with lowest hamming distance
[sortedValues,sortIndex] = sort(D,'ascend');
maxIndex = sortIndex(2:6);
figure(1)
plot(D)
days = ones(288,1)';
for day=maxIndex
    days = [days; house_state((day-1)*sample+1:day*sample)];
end
figure(2);
plot(days');
%}
%% predict day based on past day
%{
accu = 0;
for day=2:59
time = 90; % corresponds to 7:30 AM as well as offset
day_l = (day-1)*sample+1+time;
% disp(datestr(t_range(day_l)))
% disp(day)
day_states = house_state(day_l-sample+1:day_l);
house_s = house_state;
%make hamming distance max
house_s(day_l-sample+1:day_l+sample) = 3*ones(288*2,1);
for day=1:59
    D(day) = pdist([day_states;house_s((day-1)*sample+1+time:day*sample+time)],'hamming');
end
% figure(1)
% plot(D)
%take smallest hamming distance days
[sortedValues,sortIndex] = sort(D,'ascend');
maxIndex = sortIndex(2:6); %add one to get the day after which is supposed to be today.
days = ones(288,1)';
for day=maxIndex
    days = [days; house_state((day-1)*sample+1:day*sample)];
end
days = days';
% figure;
% subplot(2,1,1), plot(days)
% subplot(2,1,2), plot(house_state(day_l+1:day_l+sample))
days = days(:,2:end);
days = mode(days');
accu = accu + 100*(1-pdist([day_states;days],'hamming'));
% disp(sprintf('Accuracy:%2.1f %%',100*(1-pdist([day_states;days],'hamming'))))
end
disp(sprintf('Past Day Prediction Accuracy (average):%f %2.1f',accu/58))
%}
%% predict day based on start

