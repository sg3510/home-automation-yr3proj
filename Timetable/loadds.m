%% Load data from file

if exist('raw_DS', 'var') == 0
    clear all;

    fhandle = fopen('data2', 'r');
    raw_DS = textscan(fhandle, '%s %s %s %s %s %s');
    fclose(fhandle);
end
disp('done till here')
%% Create processable datasets
if exist('act_DS', 'var') == 0
    n_data = size(raw_DS{1},1);
    DS = datenum(strcat(raw_DS{1}(:),raw_DS{2}(:)), 'yyyy-mm-ddHH:MM:SS');
    DS = [DS zeros(n_data,1) strcmpi(raw_DS{4}, 'ON')]; %zeros(n_data,2)];

    sensorlist = unique(raw_DS{3});
    n_sensors = size(sensorlist,1);

    for i = 1:n_data
        DS(i,2) = find(strcmpi(sensorlist, raw_DS{3}{i}));
    end
    
    activitylist = unique(raw_DS{5});
    n_activities = size(activitylist,1);
    
    index = find(strcmpi(raw_DS{5},'') == 0);
    act_DS = [DS(index,1) zeros(size(index)) strcmpi(raw_DS{6}(index),'begin')];
    n_actdata = size(act_DS,1);
    
    for i = 1:n_actdata
        act_DS(i,2) = find(strcmpi(activitylist, raw_DS{5}{index(i)}));
    end    
end
disp('done till there')
% Extract sensor numbers corresponding to events.
%{
    Time of day, 4 hour granularity
    Total number of sensro firings in dT
    Binary features indicating [indv] sensor firings in interval dT
    
    Features calc'd every 5 minutes
    Markov state transition probabilities and emission probabilities
        are both calculated using frequency counting.
    Generative Gaussian model for P(xt|yt) to smooth prob. dist. of feature
        ii
%}

%% Generate features from dataset
% For now, just do this for one day.
%start_time = datenum(2009, 4, 3);
%end_time = datenum(2009, 4, 4);

start_time = ceil(DS(1)); %+7
end_time = floor(DS(end,1))-1; %datenum(start_time + 60); 
t_diff = datenum(0,0,0,0,5,0) - datenum(0,0,0,0,0,0);   %5 mins diff

t_range = start_time:t_diff:end_time-t_diff;
n_samples = size(t_range,2);

% firings - binary, indicates whether a sensor was active in t_diff
% total_firings - counts the total number of firings in t_diff
% occupancy - how many people are in the house?
firings = zeros(n_samples, n_sensors);
total_firings = zeros(size(t_range));
occupancy = zeros(size(t_range));
sleep = zeros(n_samples, 3); %1 column for each of the residents, 3rd for "house"

leave_act = find(strcmp('Leave_Home', activitylist));
arrive_act = find(strcmp('Enter_Home', activitylist));
r1sleep_act = find(strcmp('R1_Sleeping_in_Bed', activitylist));
r2sleep_act = find(strcmp('R2_Sleeping_in_Bed', activitylist));

occ_state = 1;  %Assume the house is initially occupied
sleepstart = 22;%The hours at which it's deemed acceptable to start
sleepend = 8;   %and finish sleeping.
sleep_state = [0 0];

wb = waitbar(0,'Initializing...');
for i = 1:n_samples
    prog = i/n_samples;
    waitbar(prog,wb,'Loading Data');
    samples = DS(find(DS(:,1)>=t_range(i) & DS(:,1)<t_range(i)+t_diff),:);
    total_firings(i) = sum(samples(:,3));
    firings(i,unique(samples(samples(:,3)==1,2))) = 1; %22/5/13 added == 1
    
    act_samples = act_DS(find(act_DS(:,1)>=t_range(i) ...
        & act_DS(:,1)<(t_range(i)+t_diff)),:);
    act_samples = act_samples(find(act_samples(:,2) ~= leave_act & ...
        act_samples(:,2) ~= arrive_act | act_samples(:,3) == 0),:); %filter

    %Occupancy vector------------------------------------
    %Jesus fucking Christ this is terrible.
    
    %{
    occ_acts = act_DS(find(act_DS(:,1)>=t_range(i) ...
        & act_DS(:,1)<(t_range(i)+t_diff) & act_DS(:,3) == 0 ),:);
    cur_occ = cur_occ + sum(occ_acts(:,2) == arrive_act) ...
        - sum(occ_acts(:,2) == leave_act);
    occupancy(i) = cur_occ;   
    %}
    
    depts = act_samples(find(act_samples(:,2) == leave_act ...
        & act_samples(:,3) == 0),:);
    arrvs = act_samples(find(act_samples(:,2) == arrive_act ...
        & act_samples(:,3) == 0),:);

    if occ_state == 1 && ~isempty(depts) && act_samples(end,1) > depts(end)
        occ_state = -1; %indeterminate.
    elseif occ_state == -1
        if ~isempty(act_samples)
            %Substitution rules
            if ~isempty(arrvs) && ~isempty(depts) && ...
                (depts(1) < arrvs(1) || depts(1) > act_samples(1))
                occupancy(occupancy == -1) = 1;
            elseif ~isempty(arrvs) && act_samples(1) < arrvs(1)
                occupancy(occupancy == -1) = 1;
            elseif ~isempty(arrvs) && arrvs(1) == act_samples(1)
                occupancy(occupancy == -1) = 0;
            end
            
            %Rules for transition
            if ~isempty(depts) && depts(end) < act_samples(end,1)
                occ_state = 1;
            elseif ~isempty(arrvs) && isempty(depts)
                occ_state = 1;
            elseif isempty(arrvs) && ~isempty(depts)
                occ_state = 0;
            end
        end
    elseif occ_state == 0
        if ~isempty(arrvs)
            if isempty(depts)
                occ_state = 1;
            else
                occ_state = -1;
            end
        elseif ~isempty(depts)
            sprintf('An error has occurred...');
        end
    end
    occupancy(i) = occ_state;
    %---------------------------------------------------------------------
    %Sleep vectors
    r1sleep = act_samples(act_samples(:,2) == r1sleep_act,:);
    r2sleep = act_samples(act_samples(:,2) == r2sleep_act,:);
    
    if ~isempty(r1sleep)
        sleep_state(1) = r1sleep(end,3);
    end
    if ~isempty(r2sleep)
        sleep_state(2) = r2sleep(end,3);
    end
    
    h = (t_range(i)-floor(t_range(i)))*24;
    sleep(i,:) = [sleep_state ...
        prod(sleep_state)|any(sleep_state)*(h>sleepstart & h<sleepend)];  
end
close(h)
%Zero out the last region of uncertainty....
occupancy(occupancy == -1) = 0;

%Overall house state
house_state = occupancy;
house_state(sleep(:,3) == 1) = 2;


