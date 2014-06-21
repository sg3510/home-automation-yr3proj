import csv
with open('c_dat.csv', 'rb') as csvfile:
	dat = csv.reader(csvfile, delimiter=',', quotechar='')
	for row in dat
		print ', '.join(row)