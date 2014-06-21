clear;
clc;
load usualday
house_state=num;
temp = 3;
count = 0;
count_l = 0;
for i=1:length(house_state)
    if (house_state(i) == temp)
        count=count+1;
    else
        count_l = [count_l count];
        count=0;
        temp= house_state(i);
    end
end
count_l = count_l(2:end);
mean(count_l)/96*24