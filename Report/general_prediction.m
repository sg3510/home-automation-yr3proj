clc;
clear all;
clear;
close all;
%load data
% load house_data
%{
load housestate.mat
clear t_mod
clear total_firings
clear p_state
%}
%%{
%setup variables
load usualday
t_range = dat;
house_state = num';
clear num;
clear dat;
%}
%load threemonths
% set all constants
day_samples = 96;%288
ratio = 288/day_samples; %ratio to 288 - used if non-5minute intervals
sample_p = 144/ratio; %half a day
sample_f = 144/ratio; %half a day

%% day prediction

%set time offset
time_o = 90/ratio; %equivalent to 7:30 AM

%store all data in a 3d array for optimisation
acc_3d = [1 1 1];

day_to_compare = 4; % days to perform a mode on
days = 2:length(house_state)/day_samples-1; % days to cycle through
for len=day_to_compare
    for sample_p = round(day_samples*(1:40)/40)
        for sample_f = round(day_samples*(1:40)/40)%[36 72 144]
            %set timing variable
            acc = 0;
            for day=days
                day_index = time_o+(day-1)*day_samples+1; %day with time offset
                %store data for future - i.e. to check for accuracy
                day_f = house_state(day_index:day_index+sample_f-1);
                %store data from past to compute Hamming distances
                day_p = house_state(day_index-sample_p:day_index-1);
                %prepare data on states - removing excess data
                states = house_state;
                states(day_index-sample_p:day_index+sample_f-1) = 3; %set to 3 as this is impossible state
                %loop through each day to calculate hamming distance of each
                %day relative to each other
                for i=1:length(states)/day_samples-1
        %                 disp(sprintf('Day: %d',i))
        %                 disp(datestr(t_range(i*day_samples-sample_p+time_o+1)))
        %                 disp(datestr(t_range(i*day_samples+time_o)))
                    comp = [day_p;states(i*day_samples-sample_p+time_o+1:i*day_samples+time_o)];
                    D(i) = pdist(comp,'hamming');
        %                 figure;
        %                 subplot(2,1,1), plot(states(i*day_samples-sample_p+time_o+1:i*day_samples+time_o))
        %                 subplot(2,1,2), plot(day_p)
        %                 title(sprintf('Day:%d Hamming distance:%.2f',i,D(i)))
                end
                %get minimum values of D with their indexes
                [sortedValues,sortIndex] = sort(D,'ascend');
                maxIndex = sortIndex(1:len); %get minimum hamming distance days
                %make sure not to go out of range
                if any(maxIndex == length(states)/day_samples-1)
                    index = find(maxIndex==length(states)/day_samples-1);
                    maxIndex(index) = sortIndex(len+1);
                end
                %days_hd stores all the days with the smallest hamming distance
                days_hd = ones(sample_f,len)';
                for i=maxIndex
        %                 disp(datestr(t_range(i*day_samples+time_o+1)))
        %                 disp(datestr(t_range(i*day_samples+time_o+sample_f)))
                    days_hd = [days_hd ;house_state(i*day_samples+time_o+1:i*day_samples+time_o+sample_f)];
                end
                %strip first value
                days_hd = days_hd(2:end,:)';
                days_hd = mode(days_hd');
                    days_hd(days_hd==2) = 1;
                    day_f(day_f==2) = 1;
                %calc accuracy
                acc = [acc 100*(1-pdist([days_hd; day_f],'hamming'))];
        %             figure;
        %             subplot(2,1,1), plot(days_hd)
        %             subplot(2,1,2), plot(day_f)
            end
            acc = acc(2:end);
            %print data
            fprintf('Acc Avg:%2.1f%% Std.Dev.:%2.f%% w/day len=%d s_p=%d s_f=%d\n',mean(acc),std(acc),len,sample_p,sample_f);
            acc_3d = [acc_3d;sample_p,sample_f,mean(acc)];
        end
    end
end
%store in variable to allow 3d plot
acc_3d = acc_3d(2:end,:);
