#import all used modules
import MySQLdb
import gc
import numpy as np
import scipy as sp
from scipy import spatial
from scipy import stats
from scipy import signal
import math
import sys
import time
from datetime import datetime, timedelta


#from past data predicts the expect day states
#in: day present states
#out: expected future states
def get_future_samples(day_p):
	#initialize distance array
	D = np.array([])
	#loop through each day finding the closest days in terms of hamming distance
	for i in range(1,days-2):
		#get array to compare
		comp =  np.array([day_p,house_state[i*day_samples-sample_p+time_o:i*day_samples+time_o]])[:,:]
		#store hamming distance in D
		D = np.append(D,sp.spatial.distance.pdist(comp,'hamming'))
	#sort by index
	sortIndex =  np.argsort(D)
	days_hd = []
	#get days to compare
	for i in sortIndex[0:day_to_compare]:
		days_hd.append(house_state[i*day_samples+time_o:i*day_samples+time_o+sample_f])
	#convert to array
	days_hd = np.array(days_hd)
	#get the mode
	days_hd = sp.stats.mode(days_hd)[0][0]
	return days_hd

#custom discrete decimating array function
def decimate_array(x, q):
	x_out = []
	if not isinstance(q, int):
		raise TypeError("q must be an integer")
	if len(x)%q != 0:
		raise NameError("Cannot decimate!")
	for i in range(0,len(x)/q):
		x_out.append(int(sp.stats.mode(x[i*q:(i+1)*q])[0][0]))
	return x_out
#custom array flattening function
def flatten_array(x):
	y = []
	for i in range(0,len(x)):
		y.append(x[i][0])
	return y


#define constants
day_samples = 288
ratio = 288/day_samples
#samples to regress on
sample_p = 144/ratio
#samples to look into the future of
sample_f = 288/ratio
day_to_compare = 4 # days to perform a mode on
#get minutes
minutes_sample = int(24*60/day_samples);




print "Welcome to Seb Grubb's (c) 2013 TimeTable filler & Predictor"
print "************************************************"
print '''
 .__                           __.
  \ `\~~---..---~~~~~~--.---~~| /   
   `~-.   `                   .~         _____ 
       ~.                .--~~    .---~~~    /
        / .-.      .-.      |  <~~        __/
       |  |_|      |_|       \  \     .--/
      /-.      -       .-.    |  \_   \_
      \-'   -..-..-    `-'    |    \__  \_ 
       `.                     |     _/  _/
         ~-                .,-\   _/  _/
        /                 -~~~~\ /_  /_
       |               /   |    \  \_  \_ 
       |   /          /   /      | _/  _/
       |  |          |   /    . -|/  _/ 
       )__/           \_/    -~~~|  /
         \                      /  \\
          |           |        /_---/
          \    .______|      ./
          (   /        \    /        
          |--|          /__/
'''
print "Legal notice: all incorrect future predictions are not \n\tthe responsibility of AutoHome"
#connect to sql and get all relevant data
db = MySQLdb.connect(host="127.0.0.1", port=3306, user="root", passwd="London123", db="Data")
cursor = db.cursor()
#get data
cursor.execute("SELECT State FROM H_State")
row = cursor.fetchall()
house_state = flatten_array(np.asarray(row))
#get data
cursor.execute("SELECT Timestamp FROM H_State")
date = np.asarray(cursor.fetchall())
#close SQL connection
db.close()
gc.collect()

#get date of last sample
date_object = datetime.fromtimestamp(int(date[len(date)-1][0])/1000.0)

#prepare variables needed
print "preparing past data",
days = int(math.ceil(len(house_state)/float(day_samples))) -1
print ".",
day_index = (days)*day_samples
print ".",
time_o = len(house_state)-day_index
print "."

#store past samples
print "processing data",
day_p = house_state[day_index-sample_p:day_index]
#get estimated future states
day_f = get_future_samples(day_p) #stores future predicted values
day_predict = house_state[day_index:day_index+time_o]#stores the values of a day to fill in.
print ".",
#convert to correct types
day_predict = np.asarray(day_predict)
day_f =  np.asarray(day_f).astype(int)
print ".",
#append to old data
day_predict = np.append(day_predict,day_f)
day_predict = day_predict[0:day_samples]
day_predict = decimate_array(day_predict,12)
print "."

#prepare to send to sql
#0 - away, 1 - in, 2 - asleep - my data
#0 - asleep, 1 - in , 2 -out
for i,v in enumerate(day_predict):
	if v==0:
		day_predict[i] = 2
	if v==2:
		day_predict[i] = 0
#send to mySQL
print "sending to mySQL",
#store in correctly named table
day_index_name = "Auto_%s" % ((date_object+timedelta(minutes=144*5)).strftime('%A'))
print ".",
db = MySQLdb.connect(host="127.0.0.1", port=3306, user="root", passwd="London123", db="House_states")
cursor = db.cursor()
print ".",
for i,v in enumerate(day_predict):
	command = "UPDATE `%s`\nSET state=%d\nWHERE Hour=%d;" % (day_index_name,v,i)
	cursor.execute(command)
print "."


#process other days in the week
day_list = range(0,(date_object+timedelta(minutes=144*5)).weekday())
day_list.extend(range((date_object+timedelta(minutes=144*5)).weekday()+1,7))
#loop through every last 4 days of the week and take the mode of each day
for d in day_list:
	iteration = 0
	day_states_data = []
	for i in range(days-1,-1,-1):
		if datetime.fromtimestamp(int(date[i*day_samples][0])/1000.0).weekday() == d:
			iteration+=1
			day_states_data.append(house_state[i*day_samples:(i+1)*day_samples])
		if iteration >= day_to_compare:
			break
	day_index_name = "Auto_%s" % (datetime.fromtimestamp(int(date[i*day_samples][0])/1000.0).strftime('%A'))
	day_states_data = decimate_array(np.asarray(sp.stats.mode(day_states_data)[0][0]).astype(int),12)
	#invert 2 and 0
	for i,v in enumerate(day_states_data):
		if v==0:
			day_states_data[i] = 2
		if v==2:
			day_states_data[i] = 0
	for i,v in enumerate(day_states_data):
		command = "UPDATE `%s`\nSET state=%d\nWHERE Hour=%d;" % (day_index_name,v,i)
		cursor.execute(command)
#close SQL connection
db.close()
gc.collect()


print'''                __....___ ,  .
            _.-~ __...--~~ ~/\\
           / /  /          |  |
          | |  |            \\  \\
    __ _..---..-~\\           |  | 
   |  ~  .-~-.    \\-.__      /  | 
   /     \\.-~        .-~-._/   // 
  |/-. <| __  .-\\    \\     \\_ //  
  || o\\   \\/ /o  |    ~-.-~  \\/         
 /  ~~        ~~              |      
 \\__         ___--/   \\  _-~-  \\ 
  / ~~--.--~~    /     |/   __  |
 |/\\ \\          |_~|   /    \\|  |
 |\\/  \\__       /_-   /\\        |
 |_ __| |`~-.__|_ _ _/  \\ _ _ _/
 ' '  ' ' ''   ' ' '     ' ` `
'''