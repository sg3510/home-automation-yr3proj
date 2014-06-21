function [env_data] = load_env_data(directory)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
env_data = [];

listing = dir(directory);
numb_files = length(listing);

for j = 3:numb_files
    str = [directory '/' listing(j).name];
    temp = csvread(str);
    env_data = [env_data; temp];
end

end
