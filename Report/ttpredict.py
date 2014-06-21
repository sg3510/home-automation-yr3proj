import csv
import numpy as np
import scipy as sp
from scipy import spatial
from scipy import stats
from datetime import datetime

#define constants
day_samples = 96
ratio = 288/day_samples
sample_p = 144/ratio
sample_f = 144/ratio
day_to_compare = 4 # days to perform a mode on
#initialize arrays
house_state = []
date = []


with open('c_dat.csv', 'rb') as csvfile:
	dat = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in dat:
		house_state.append(row[0])
		date.append(row[1])
d = datetime.now()
#d = datetime(2012, 3, 23, 12, 00, 00, 173504)
time_o = int((d.hour + d.minute/60.0)*day_samples/(24.0))
day = 27
#get day_index
day_index = time_o+(day-1)*day_samples
print "Date is %s at index %d" % (date[day_index],day_index)
day_f = np.array(house_state[day_index:day_index+sample_f-1])
day_p = np.array(house_state[day_index-sample_p:day_index-1])
states = np.array(house_state)    
states[day_index-sample_p:day_index+sample_f-1] = 3 #set to 3 as this is impossible state
D = np.array([])
for i in range(0,len(states)/day_samples-2):
	comp = np.array([day_p,states[i*day_samples-sample_p+time_o:i*day_samples+time_o-1]])
	#print "%d:from %s to %s" % (i,date[i*day_samples-sample_p+time_o],date[i*day_samples+time_o-1])
	#print sp.spatial.distance.pdist(comp,'hamming')
	D = np.append(D,sp.spatial.distance.pdist(comp,'hamming'))
sortIndex =  np.argsort(D)
#print D[sortIndex[0:day_to_compare]]
#sortedValues,sortIndex = sort(D,'ascend');
days_hd = []
for i in sortIndex[0:day_to_compare]:
	days_hd.append(house_state[i*day_samples+time_o:i*day_samples+time_o+sample_f-1])
days_hd = np.array(days_hd)	
days_hd = sp.stats.mode(days_hd)[0][0]
days_hd= np.array([days_hd,day_f])
100*(1-sp.spatial.distance.pdist(days_hd,'hamming'))
print "You are %d%% correct" % int(100.0*(1-sp.spatial.distance.pdist(days_hd,'hamming')))
