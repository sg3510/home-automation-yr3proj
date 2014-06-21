#!/usr/bin/python
import MySQLdb
import math
import random
import sqlite3
import time
from functions import *
from websocket import create_connection
from datetime import datetime
from scipy import *
from numpy import *



#GPR_RUN:
#Function reads in training data, trained hyperparameters and inverse C and uses them to calculate
#a thermostat estimate. This is then returned to be sent via websockets back to the embed module.
def active_TS(x):
	#print("active_TS function entered, calculating TS value using Gaussian Process Regression...")
	db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","GPR_Training_Data" )
	cur = db.cursor()
	#Read in trained hyperparameters to theta from SQL database
	cur.execute("SELECT * FROM trained_theta_values")
	row = cur.fetchall()
	numb_hp = len(row[0])
	theta = zeros(numb_hp)
	for j in range(numb_hp):
        	theta[j] = row[0][j]
	#Read in inv_C from SQL database
	cur.execute("SELECT * FROM inv_C")
	rows = cur.fetchall()
	numrows = int(math.sqrt(len(rows)))
	inv_C  = ones((numrows, numrows))
	for j in range(numrows):
        	for i in range(numrows):
                	inv_C[j][i] = rows[i][2]
	#Import the training data required in calculation
	cur.execute("SELECT * FROM training_data")
	rows = cur.fetchall()
	numrows = len(rows)
	numcols = len(rows[0])
	X = zeros((numrows, (numcols-1)))
	y = zeros((numrows, 1))
	for j in range((numrows)):
			y[j] = rows[j][0]
	for j in range(numrows):
		for i in range(1, numcols):	
			X[j][(i-1)] = rows[j][i]
			
	#Normalize Data
	X, x = normalize_input(X, x)
	
	#Calculate K_star and then calculate expected TS value given input x
	K_star = calc_K_star(x, X, theta)
	m = len(X[0])
	M = zeros((m,m))
	for i in range(m):
		M[i][i] = 1/(theta[i+2]*theta[i+2])
	K_starstar = cov_func(x, x, M, theta[0])
	TS = calc_E_TS(inv_C, K_star, y)
	TS_conf = K_starstar - dot(K_star, dot(inv_C, K_star.transpose()))
	print "Predicted TS: %d, Confidence in prediction: %d" % (TS, TS_conf)
	return TS, TS_conf



		
#Function is necessary as for practical reasons because the training data set cannot be allowed to grow indefinitly. Instead 
#new predicted values have to be swapped in for older training examples. Before they can be considered however new training
#data points have to be cross validated agains the user input log to make sure they were not over ridden. This function takes each
#entry in the user log in turn and removes any entry in potential data entry that is within a certain time the user entry. All potential
#training data points that pass this test are passed into temp to be inserted into the training data set by insert_td.
def validate_TD(sig_time):
	print("Validating potential new training data against user log...")
	db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","GPR_Training_Data")
	cur = db.cursor()
	
	clean_time = 86400
	
	#Import the potential training data
	cur.execute("SELECT * FROM potential_new_training_data")
	rows = cur.fetchall()
	if(len(rows) > 0):
		ptd = zeros((len(rows), len(rows[0])))
		for j in range(len(rows)):
			for i in range(len(rows[0])):
				ptd[j][i] = rows[j][i]
		
		#Import the user log
		cur.execute("SELECT * FROM user_TS_input_log")
		rows = cur.fetchall()
		user_log = zeros((len(rows), len(rows[0])))
		for j in range(len(rows)):
			for i in range(len(rows[0])):
				user_log[j][i] = rows[j][i]
		
		#PROCESS
		for j in range(len(ptd)):
			val_pass = True
			for i in range(len(user_log)):
				time_diff = (user_log[i][0] - ptd[j][6])
				print time_diff
				if(time_diff < 0):
					val_pass = True
				elif((time_diff > 0) & (time_diff<(sig_time*60))):
					val_pass = False
					break
				elif(time_diff > (sig_time*60)):
					val_pass = True
			print("Validation test result: ", val_pass)
			
			if(val_pass == True):			
				print("Entry in potential new training data with following id passed: ", j)
				command = "INSERT INTO temp (thermostat_setting, external_temp, external_humidity, internal_humidity, wind_speed, wind_direction) VALUES ('%s', '%s', '%s', '%s', '%s', '%s')"%(ptd[j][0], ptd[j][1], ptd[j][2], ptd[j][3], ptd[j][4], ptd[j][5])
				print("Sending following command to database to add entry to table 'temp': ", command)
				cur.execute(command)
		
		print("Operation done, all entries that passed vaildation test moved to 'temp' for insertion into 'training data'.")
		cur.execute("TRUNCATE TABLE potential_new_training_data")
		db.close()
	
	print("Removing old data entries in user log")
	#Get current timestamp (ms)
	current_timestamp = int(time.time())
	time_bound = current_timestamp - clean_time
	command = "DELETE FROM user_TS_input_log WHERE time_stamp < '%d'" % int(time_bound)
	cur.execute(command)
	print("Validation process complete")
	
	
#Function is necessary as for practical reasons because the training data set cannot be allowed to grow indefinitly. Instead 
#new predicted values have to be swapped in for older training examples. This function takes all the data points that pass
#the validation test and adds as many of them as it can to the training set till capacity is reached. After this new training examples
#are swapped in by removing the training points that are closest in the feature space.
def insert_new_td(desired_training_set_size):
	print("INSERTING NEW DATA INTO THE TRAINING DATA SET...")
	db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","GPR_Training_Data")
	cur = db.cursor()
	
	#Import the number of new entries
	cur.execute("SELECT COUNT(*) FROM temp")
	result=cur.fetchone()
	numb_new_entries = result[0]
	print("Number of new entries: ", numb_new_entries)
	
	#Import new data to be added to training data if it exists
	if(numb_new_entries > 0):
		cur.execute("SELECT * FROM temp")
		rows = cur.fetchall()
		temp = zeros((len(rows), len(rows[0])))
		for j in range(len(rows)):
			for i in range(len(rows[0])):
				temp[j][i] = rows[j][i]
		print("Data to be inserted: ", temp)
	
		#Import the current training data
		cur.execute("SELECT * FROM training_data")
		rows = cur.fetchall()
		numrows = len(rows)    
		numcols = len(rows[0])
		X = zeros((numrows, numcols))
		for j in range((numrows)):
			for i in range(numcols):	
				X[j][i] = rows[j][i]
		X_norm = normalize_data(X)#Normalize to ensure each feature has equal significance in the sum
		numb_training_points = len(X)
		
		
		if(numb_new_entries <= abs(desired_training_set_size - numb_training_points)):#Automatically insert all points if there is enough capacity in the training set.
			print("TRAINING SET NOT AT CAPACITY - INSERTING ALL NEW POINTS")
			for j in range(len(temp)):
				cur.execute("INSERT INTO training_data (thermostat_setting, external_temp, external_humidity, internal_humidity, wind_speed, wind_direction) VALUES (%s, %s, %s, %s, %s, %s)", (temp[j][0], temp[j][1], temp[j][2], temp[j][3], temp[j][4], temp[j][5]))
				
		else:
			print("TRAINING SET NEAR CAPACITY - REPLACING OBSELETE POINTS")#Automatically insert new points till capacity is reached
			numb_auto_insert = abs(desired_training_set_size - numb_training_points)
			print("Number to auto-insert into the training data set: ", numb_auto_insert)
			if(numb_auto_insert > 0):
				for j in range(numb_auto_insert):
					cur.execute("INSERT INTO training_data (thermostat_setting, external_temp, external_humidity, internal_humidity, wind_speed, wind_direction) VALUES (%s, %s, %s, %s, %s, %s)", (temp[j][0], temp[j][1], temp[j][2], temp[j][3], temp[j][4], temp[j][5]))
				
			for j in range(numb_auto_insert, len(temp)):#Swap in new training points for old ones that are closest to them in the feature space.
				similarities = zeros(len(X))
				for i in range(len(X_norm)):
					similarities[i] = calc_similarity(temp[j], X_norm[i])	
				index_replace = similarities.argmax()
				command = "UPDATE `training_data` SET `thermostat_setting`='%s', `external_temp`='%s', `external_humidity`='%s', `internal_humidity`='%s', `wind_speed`='%s', `wind_direction`='%s' WHERE `thermostat_setting`='%s' AND `external_temp`='%s' AND `external_humidity`='%s' AND `internal_humidity`='%s' AND `wind_speed`='%s' AND `wind_direction`='%s';" % (temp[j][0], temp[j][1], temp[j][2], temp[j][3], temp[j][4], temp[j][5], X[index_replace][0], X[index_replace][1], X[index_replace][2], X[index_replace][3], X[index_replace][4], X[index_replace][5])
				cur.execute(command)
				print("Sent following command to sequel database: ", command)
		cur.execute("TRUNCATE TABLE temp")
	db.close()
	

#GPR TRAIN:	
#Function trains the GPR by optimizing hyperparameters using gradient descent.	
def train_GPR(alpha, max_iters, error_level, init_lower, init_upper, num_init):
	print("TRAINING GPR ALGORITHM")
	#READ IN TRAINING DATA
	# Open database connection
	db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","GPR_Training_Data" )
	# prepare a cursor object using cursor() method
	cur = db.cursor()
	#Read in training data from SQL database
	cur.execute("SELECT * FROM training_data")
	rows = cur.fetchall()
	# disconnect from server
	db.close()
	
	#PUT DATA INTO MATRICES
	#Find the number of rows (data points) and columns (Features
	numrows = len(rows)    
	numcols = len(rows[0])
	#Process data into input feature data matrix X and actual outputs associated y
	X = zeros((numrows, (numcols-1)))
	y = zeros((numrows, 1))
	for j in range((numrows)):
			y[j] = rows[j][0]

	for j in range(numrows):
		for i in range(1, numcols):	
			X[j][(i-1)] = rows[j][i]
			
	#NORMALIZE DATA
	X = normalize_data(X)

	#OPTIMISE AND DEFINE HYPERPARAMETERS USING GRADIENT DESCENT
	#Number of Input features
	m = len(X[0])
	#Number of data points 
	n = len(y)
	
	theta_init = random.uniform(init_lower,init_upper,(m+2))
	theta, successful, nlml, nlml_history, exit_iters = gradient_descent(alpha, theta_init, X, y, error_level, max_iters)
	
	#Carry out gradient descenet multiple times as with different initialisations may get different local minima and therefore 
	#worse or better performance. Select the set of hyperparameters that gives the lowest NLML. 
	for j in range(num_init):
		print("j is: ", j)
		#Define and initialise theta, note is a row vector
		theta_init = random.uniform(init_lower,init_upper,(m+2))
		print("Initial Theta values: ", theta_init)
		#Use gradient descent to find optimized parameters
		theta_test, successful, nlml_test, nlml_history, exit_iters = gradient_descent(alpha, theta_init, X, y, error_level, max_iters)
		print("Initial nlml:", nlml_history[0])
		print("Final nlml: ", nlml)
		if(nlml_test < nlml):
			theta = theta_test
			nlml = nlml_test
	print("Theta optimized: ", theta)
	
	
	#CALCULATE MODEL VALUES
	#Calculate K with optimized hyperparameters
	K = calc_K(X, theta)
	#Calculate 'C'
	C = calc_C(K, theta[1])
	#Calculate inverse of C
	inv_C = linalg.inv(C)
	#OUTPUT INVERSE C AND THETA VALUES TO DATABASE TABLES 
	# Open database connection
	db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","GPR_Training_Data" )
	cursor = db.cursor()
	#Empty table of obselete theta values
	cursor.execute("TRUNCATE TABLE trained_theta_values")
	#Insert trained theta values into the database
	cursor.execute("INSERT INTO trained_theta_values (signal_power, noise_power, external_temp, external_humidity, internal_humidity, wind_speed, wind_direction) VALUES (%s, %s, %s, %s, %s, %s, %s)", (theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6]))
	#Empty table of obselete inv_C values
	cursor.execute("TRUNCATE TABLE inv_C")
	#Insert inv_C values into the database
	for j in range(n):
		for i in range(n):
			cursor.execute("INSERT INTO inv_C (row_number, column_number, value) VALUES (%s, %s, %s)", (j, i, inv_C[j][i]))
	db.close()
	print("GPR Algorithm trained")
	

#Function is called everytime a thermostat setting is requested.
#Description: Function reads in input data and recent user input tables. If the predicted state is occupied, and there has been a user input (i.e. a user input since the last call of this function) then 
#the user input is used, otherwise former inputs from the user (found in the user log table) are used if they are within a certain  time range of the current timestamp. Else a TS value is generated by 
#the GPR algorithm. If the state is 'empty' or 'asleep' default settings are adopted.	
def get_TS(internal_humidity):
	#DEFINE VARIABLES USED IN PROGRAM
	#k is time in minutes that a user update holds for
	alpha = 0.0000001 
	error_level = 1 
	max_iters = 100
	numb_init = 10
	k = 60
	unoccupied_temp = 2
	desired_training_set_size = 500
	#Initialise TS
	TS = 0
	
	#GET HYPERPARAMETER INITIALISATION RANGE
	db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","algorithm_testing" )
	cur = db.cursor()
	cur.execute("SELECT * FROM hyperparameter_range")
	row = cur.fetchall()
	lb = row[0][0]
	ub = row[0][1]
	db.close()

	#GET DATA REQUIRED
	db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","Data" )
	cur = db.cursor()
	#Import user settings
	cur.execute("SELECT * FROM limits")
	row = cur.fetchall()
	max_temp = row[0][0]
	min_temp = row[0][1]
	sleeping_temp = row[0][2]
	print("IMPORTED USER SETTINGS: ")
	print("max_temp is: ", max_temp)
	print("min_temp is: ", min_temp)
	print("sleeping_temp is: ", sleeping_temp)

	#Import the input for this call
	x, predicted_state, current_timestamp = get_input(internal_humidity)
	
	print("DATA TO BE USED TO GENERATE TS IF NO USER INPUT")
	print("New input data entry: ",x)
	print("State: ",predicted_state)
	print("Timestamp: ", current_timestamp)

	print("ASCERTAINING IF THERE HAS BEEN A USER UPDATE...")
	#Ascertain if there has been a recent user update
	cur.execute("SELECT * FROM user_update")
	rows = cur.fetchall()
	numb_recent_updates = len(rows)
	print("Number of user updates in last k minutes: ", numb_recent_updates)

	if(numb_recent_updates > 0):
		new_user_update = True
		user_update = zeros((len(rows), len(rows[0])))
		for j in range(len(rows)):
			for i in range(len(rows[0])):	
				user_update[j][i] = rows[j][i]
			print("user update data: ", user_update)

	else:
		new_user_update = False

		
	print("State of user update: ", new_user_update)
	db.close()

	#PROCESS
	#If the user has made a change since the last call then use this
	if(new_user_update == True):
		print("USER UPDATE DETECTED - USER INPUT PATH TAKEN")
		if(predicted_state == 0):
			print("State: Unoccupied, setting TS to unoccupied setting...")
			TS = unoccupied_temp
		elif(predicted_state == 1):
			print("State: Occupied, setting TS to user setting...")
			TS = user_update[(numb_recent_updates-1)][0]
		elif(predicted_state == 2):
			print("State: Asleep, setting TS to sleeping setting...")
			TS = sleeping_temp
		else:
			print("Error: 'predicted_state' value not valid:", predicted_state)
		
		
		if(TS > max_temp):
			TS = max_temp
			
		elif(TS < min_temp):
			TS = min_temp

		else:
			print("TS within boundry conditions")

		print("TS value selected",TS)
		
		#OUTPUT TS TO THE EMBED
		print("OUTPUT TO EMBED...")
		ws = create_connection("ws://ec2-54-214-164-65.us-west-2.compute.amazonaws.com:8888/ws") 
		print("Sending Thermostat Setting...")
		command = '{"type": "thermostat_control" , "setting" : %d}' % int(TS)
		print("Output to embed: ", command)
		ws.send(command)
		print("Thermostat Setting Sent with value of: ", TS)
		ws.close()
		
		
		#HANDLE DATABASE
		db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","GPR_Training_Data" )
		cur = db.cursor()
		#Upload the user input to the temp file and retrain GPR algorithm
		cur.execute("INSERT INTO temp (thermostat_setting, external_temp, external_humidity, internal_humidity, wind_speed, wind_direction) VALUES (%s, %s, %s, %s, %s, %s)", (user_update[(numb_recent_updates-1)][0], user_update[(numb_recent_updates-1)][1], user_update[(numb_recent_updates-1)][2], user_update[(numb_recent_updates-1)][3], user_update[(numb_recent_updates-1)][4], user_update[(numb_recent_updates-1)][5]))
		#Upload the user input to the user log
		command = "INSERT INTO user_TS_input_log (thermostat_setting, external_temp, external_humidity, internal_humidity, wind_speed, wind_direction, time_stamp) VALUES (%s, %s, %s, %s, %s, %s, %s)" % (user_update[(numb_recent_updates-1)][0], user_update[(numb_recent_updates-1)][1], user_update[(numb_recent_updates-1)][2], user_update[(numb_recent_updates-1)][3], user_update[(numb_recent_updates-1)][4], user_update[(numb_recent_updates-1)][5], user_update[(numb_recent_updates-1)][6])
		print("Uploading following entry into user log: ", command)
		cur.execute(command)
		#Retrain Algorithm and truncate 'temp' afterwards
		insert_new_td(desired_training_set_size)
		train_GPR(alpha, max_iters, error_level, lb, ub, numb_init)
		db.close()
		
		db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","Data" )
		cur = db.cursor()
		#Truncate user_data (user input)
		cur.execute("TRUNCATE TABLE user_update")
		#Truncate Data (input)
		cur.execute("TRUNCATE TABLE Data")
		

	#NO INPUT SINCE LAST CALL...	
	else:
		print("USER UPDATE NOT DETECTED - CHECKING FOR FORMER STILL RELEVANT USER INPUTS...")
		
		#Import user log data
		db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","GPR_Training_Data" )
		cur = db.cursor()
		cur.execute("SELECT * FROM user_TS_input_log")
		rows = cur.fetchall()
		numb_user_entries = len(rows)
		user_time_stamps = zeros(numb_user_entries)
		
		#Load in the time stamps in the user log
		for j in range(numb_user_entries):
			user_time_stamps[j] = rows[j][0]
			
		print("Numer of entries in user log: ", numb_user_entries)
		print("User Log timestamps:", user_time_stamps)
		
		if(numb_user_entries > 0):
			#Check if any entries in the user log are relevant
			id, closest_user_timestamp, min_diff = find_nearest(user_time_stamps,current_timestamp)
			if(fabs(current_timestamp - closest_user_timestamp) < (k*60)):#Note timestamp is in milliseconds, k*60*10^3 ms in k minutes
				use_user_input = True

			else:
				use_user_input = False
				
			print("Current Time Stamp: ", current_timestamp)
			print("Closest time difference to current timestamp: ", min_diff)
			
		else:
			use_user_input = False
		
		print("State of use user log input: ", use_user_input)	
			
		#Process
		if(predicted_state == 0):
			print("State: Unoccupied, setting TS to unoccupied setting")
			TS = unoccupied_temp
		elif(predicted_state == 1):
			if(use_user_input == True):
				print("Former user input still relevant, referring back to last user log entry...")
				TS = user_TS(current_timestamp)
			else:
				print("Former user entries no longer relevant, generating TS from GPR algorithm...")
				TS, TS_conf = active_TS(x)	
		elif(predicted_state == 2):
			print("State: Asleep, setting TS to sleeping setting...")
			TS = sleeping_temp
		else:
			print("Error: 'predicted_state' value not valid:")
			print predicted_state
		
		#Bound the TS setting
		if(TS > max_temp):
			TS = max_temp
			
		elif(TS < min_temp):
			TS = min_temp
		else:
			print("TS within boundry conditions")
		
		#OUTPUT TS TO THE EMBED
		print("OUTPUT TO EMBED...")
		ws = create_connection("ws://ec2-54-214-164-65.us-west-2.compute.amazonaws.com:8888/ws") 
		print("Sending Thermostat Setting...")
		command = '{"type": "thermostat_control" , "setting" : %d}' % int(TS)
		print("Output to embed: ", command)
		ws.send(command)
		print("Thermostat Setting Sent with value of: ", TS)
		ws.close()
		
		
		#HANDLE DATABASE UPDATES
		if((use_user_input == False) & (predicted_state == 1)):
			db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","GPR_Training_Data" )
			cur = db.cursor()
			#Upload input to potential new training data
			cur.execute("INSERT INTO potential_new_training_data (thermostat_setting, external_temp, external_humidity, internal_humidity, wind_speed, wind_direction, time_stamp) VALUES (%s, %s, %s, %s, %s, %s, %s)", (TS, x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], current_timestamp))
			db.close()
			
			
		
		
#Function takes all the data in the potential_training_data table, runs a validation check against the user_TS_input_log and then outputs the data entries
#that pass into the 'temp' data table these are then inserted into the training data table and the GPR retrain algorithm is re-run. The temp and potential data table are then truncated
#and all entries older than a day in the user log are also removed. Note user inputs are automatically added to the training set as soon as they are detected. Called approximatly 
#every 24 hours.		
def retrain_with_updates():
	#GET HYPERPARAMETER INITIALISATION RANGE
	db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","algorithm_testing" )
	cur = db.cursor()
	cur.execute("SELECT * FROM hyperparameter_range")
	row = cur.fetchall()
	lb = row[0][0]
	ub = row[0][1]
	db.close()

	#PARAMETER DEFINITIONS
	alpha = 0.0000001
	error_level = 1
	max_iters = 100
	numb_init = 10
	sig_time = 60
	desired_training_set_size = 500	

	#Check data against user log entries, if within a certain time distance then don't insert into the table 'temp'. Once complete truncates 'potential_training_data' and removes old entries in user log
	validate_TD(sig_time)

	#Insert data in 'temp' table into the 'training_data' table appropriatly. Once complete truncates 'temp'
	insert_new_td(desired_training_set_size) 

	#Retrain Algorithm, i.e. obtain new theta values and correlation matrix
	train_GPR(alpha, max_iters, error_level, lb, ub, numb_init)
	
	
	
#HYPERPARAMETER SEARCH:
#Different initialisations of theta lead to different local minima and therefore different performance. This function searches the NLML space and initializes 
#the hyperparameters in sub regions. It checks the performance of each region by comparing the predictions with the actual values from a test set (used to train
#the data) and a cross validation set (an independant set) and calculates the percentage error for both. The region with the best performance is then subdivided again
#until a certain initialisation range is reached. It then returns this initialisation range to the database to be used by all future 'train calls' to initialise the hyperparameters.#
#This function is simply put a range finder and gives an idea of where to look for the best minima, it is only called when there are very large changes in the data set. 	
def hyperparameter_search():
	#INITIALISE SYSTEM PARAMETERS
	alpha = 0.00001
	error_level = 1
	max_iters = 50
	num_init = 20 

	#IMPORT INPUT DATA
	db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","algorithm_testing" )
	cur = db.cursor()

	#Read in training data
	cur.execute("SELECT * FROM training_data")
	rows = cur.fetchall()
	train_d = zeros((len(rows), len(rows[0])))
	for j in range(len(rows)):
		for i in range(len(rows[0])):
			train_d[j][i] = rows[j][i] 

	#Read in test data
	cur.execute("SELECT * FROM test_data")
	rows = cur.fetchall()
	test_d = zeros((len(rows), (len(rows[0])-1)))
	TS_test = zeros((len(rows)))
	#print len(rows)
	for j in range(len(rows)):
		TS_test[j] = rows[j][0]
		for i in range(1,len(rows[0])):
			test_d[j][(i-1)] = rows[j][i] 

	lower_bound = 0
	upper_bound = 100		
	numb_divs = 10
	m = len(test_d[0])

	#Perform the grid search
	while((upper_bound - lower_bound) > 4):
		increment = (upper_bound - lower_bound)/numb_divs
		print "Considering range of initialisations %d to %d:" % (lower_bound, upper_bound)
		print "Increment is: %d" % (increment)
		lb_temp = lower_bound
		ub_temp = lb_temp + increment
		av_err = zeros((numb_divs))
		for k in range((numb_divs)):
			#TRAIN ALGORITHM:
			train_GPR(alpha, max_iters, error_level, lb_temp, ub_temp, num_init)
			#Initialise
			err = zeros((len(TS_test)))
			gpr_res = zeros((len(TS_test)))
			gpr_conf = zeros((len(TS_test)))
			for j in range(len(test_d)):
				gpr_res[j], gpr_conf[j] = active_TS(test_d[j])
				err[j] = 100*((abs(TS_test[j] - gpr_res[j]))/TS_test[j])
				av_err[k] = av_err[k] + err[j]
			
			av_err[k] = av_err[k]/(len(test_d))
			print "Error bound for division %d-%d is: %d" % (lb_temp, ub_temp, av_err[k])		
			lb_temp = ub_temp
			ub_temp = ub_temp + increment
				
		ind_min_err = np.argmin(av_err)	
		print "the index of the minimum division is: %d" % (ind_min_err)
		upper_bound = lower_bound + (ind_min_err*increment)
		lower_bound = upper_bound - increment
		print "Theta values found to be best initialised in this range are between %d - %d" % (lower_bound, upper_bound)	

			
	print "Theta values found to be best initialised in this range are between %d - %d" % (lower_bound, upper_bound)
	print("Initialised with these Boundaries we have...")

	train_GPR(alpha, max_iters, error_level, lower_bound, upper_bound,num_init)	
	err = zeros((len(TS_test)))
	gpr_res = zeros((len(TS_test)))
	gpr_conf = zeros((len(TS_test)))
	av_err = 0
	for j in range(len(test_d)):
		gpr_res[j], gpr_conf[j] = active_TS(test_d[j])
		err[j] = 100*((abs(TS_test[j] - gpr_res[j]))/TS_test[j])#Calculate error on each prediction
		av_err= av_err + err[j]
	av_err = av_err/len(test_d)#Calculate the % error in the test set

	min = np.argmin(err)
	max = np.argmax(err)

	print("TS values from test set", TS_test)
	print("TS values calculated", gpr_res)
	print("Variance of GPR values", gpr_conf)
	print("Percentage Error List:", err)
	print "Max percentage error in estimate: %d" % (err[max])
	print "Min percentage error in estimate: %d" % (err[min])
	print "Average Percentage Error in estimates: %d" % (av_err)

	cur.execute("TRUNCATE TABLE hyperparameter_range")
	cur.execute("INSERT INTO hyperparameter_range (lower_bound, upper_bound) VALUES (%s, %s)" % (lower_bound, upper_bound))  
	
	
	
	
	
	
#With insufficient data to run the state detection algorithm this function is used to generate values	
def estimate_current_state():
	x = random.randint(1,100)
	if x <= 10:
		return 0
	elif x <= 66:
		return 1
	else:
		return 2
		
		

#Function is run before 'get_TS' is called to compile an input vector		
def get_input(internal_humidity):
	db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","Data" )
	cur = db.cursor()
	#Import enviromental data from Data2
	command = "SELECT Temperature, Humidity, Wind_speed, Wind_direction FROM Data2 WHERE Time_stamp=(select MAX(Time_stamp) from Data2)"
	print("Command sent to SQL database: ", command)
	cur.execute(command)
	temp = cur.fetchone()
	print temp
	#Get the state of the house:
	state = find_state()
	#Get current Time stamp
	current_timestamp = int(time.time())/1000
	#Format Oututs:
	x = zeros((5))
	x[0] = temp[0]
	x[1] = temp[1]
	x[2] = internal_humidity
	x[3] = temp[2]
	x[4] = temp[3]
	return x, state, current_timestamp
	
	


#Function uses a logic table to select a state by comparing the predicted and detected state. It then returns this state to
#the 'get_TS' function.
def find_state():
	localtime = time.localtime(time.time())
	print "Day is: %d" % (localtime[6])
	print "Hour is: %d" % (localtime[3])

	hour = localtime[3]

	db = MySQLdb.connect("127.0.0.1","mike","ChickenKorma","House_states" )
	cur = db.cursor()

	#First need find the correct day, the data is puleld of the same table that feeds the website.
	if(localtime[6] == 0):
		command = "SELECT state FROM Monday WHERE Hour =  '%d'" % (hour)
		print("Command sent to SQL database: ", command)
		cur.execute(command)
		temp = cur.fetchone()
		pred_state = int(temp[0])
		
	elif(localtime[6] == 1):
		command = "SELECT state FROM Tuesday WHERE Hour =  '%d'" % (hour)
		print("Command sent to SQL database: ", command)
		cur.execute(command)
		temp = cur.fetchone()
		pred_state = int(temp[0])
		
	elif(localtime[6] == 2):
		command = "SELECT state FROM Wednesday WHERE Hour =  '%d'" % (hour)
		print("Command sent to SQL database: ", command)
		cur.execute(command)
		temp = cur.fetchone()
		pred_state = temp[0]
		
	elif(localtime[6] == 3):
		command = "SELECT state FROM Thursday WHERE Hour =  '%d'" % (hour)
		print("Command sent to SQL database: ", command)
		cur.execute(command)
		temp = cur.fetchone()
		pred_state = temp[0]
		
	elif(localtime[6] == 4):
		command = "SELECT state FROM Friday WHERE Hour =  '%d'" % (hour)
		print("Command sent to SQL database: ", command)
		cur.execute(command)
		temp = cur.fetchone()
		pred_state = temp[0]
		
	elif(localtime[6] == 5):
		command = "SELECT state FROM Saturday WHERE Hour =  '%d'" % (hour)
		print("Command sent to SQL database: ", command)
		cur.execute(command)
		temp = cur.fetchone()
		pred_state = temp[0]
		
	elif(localtime[6] == 6):
		command = "SELECT state FROM Sunday WHERE Hour =  '%d'" % (hour)
		print("Command sent to SQL database: ", command)
		cur.execute(command)
		temp = cur.fetchone()
		pred_state = temp[0]
	else:
		print "Day not recognized, day = '%d'" % (localtime[6])
	db.close()
		
	rt_state = estimate_current_state()
	print type(rt_state)
	print type(pred_state)
#Logic table
	if(rt_state == 0 and (pred_state == 0 or pred_state == 3)):
		state = 0
	elif(rt_state == 0 and (pred_state == 1 or pred_state == 4)):
		state = 1
	elif(rt_state == 0 and (pred_state == 2 or pred_state == 5)):
		state = 0
	elif(rt_state == 1 and (pred_state == 0 or pred_state == 3)):
		state = 1
	elif(rt_state == 1 and (pred_state == 1 or pred_state == 4)):
		state = 1
	elif(rt_state == 1 and (pred_state == 2 or pred_state == 5)):
		state = 2
	elif(rt_state == 2 and (pred_state == 0 or pred_state == 3)):
		state = 2
	elif(rt_state == 2 and (pred_state == 1 or pred_state == 4)):
		state = 1
	elif(rt_state == 2 and (pred_state == 2 or pred_state == 5)):
		state = 2
	else:
		print "Real time and predicted state not recognised, rt_state = '%d', pred_state = '%d'" % (rt_state, pred_state)
	
	return state

random.seed()	
	
	
	

