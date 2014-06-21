#import required libraries
import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import time
import json
import requests
import random
import functions as mfunc
import system_primary_functions as syspf

#set user count to 0
users = 0
#custom function to test if a string is JSON
def json_test(string):
    try:
        json.loads(string)
        return 1
    except:
        return 0
        
#handle http
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('{"type":"message","message":"Hello this is not the correct websocket address. Please add \ws."}')
#websocket handler
class WSHandler(tornado.websocket.WebSocketHandler):
	#keep track of all users currently connected
    clients = []    
    def allow_draft76(self):
        # to use same draft as mbed library
        return True
	#on connection open
    def open(self):
		#add user to user list
        self.clients.append(self)
        print 'new connection'
		#update user count
        global users
        users = users + 1
        print 'users : %d' % users
		#send welcome message
        self.write_message('{"type":"message","message":"Hello World"}')
		#log the event
        with open("tornado_ws.log", "a") as myfile:
            localtime = time.asctime( time.localtime(time.time()) )
            myfile.write('%s:new connection : %d users \r\n' % (localtime, users))
     #on message receipt
    def on_message(self, message):
		#log the event
        print 'message received %s' % message
        with open("tornado_ws.log", "a") as myfile:
            localtime = time.asctime( time.localtime(time.time()) )
            if message != '{"type":"ping"}':
                myfile.write('%s:message received %s \r\n' % (localtime,message)) 
        #test if message is JSON and decide whether to process further
		if (json_test(message) == 1):
			print 'message is JSON'
			#extract data
            data = json.loads(message)
			#do different actions based on type of message
            if data['type'] == 'house_measurement':
                print "House measurement Data"
                self.write_message('{"type":"message","message":"Sent to Mike GPR"}')
                try:
					#pass on to rest of clients if temperature data received
                    response =  '{"type":"temperature_measurement","temperature":%d}' % (data['Temperature'])
                    print response
                    for client in self.clients:
                        client.write_message(response)
                except:
					print "No temp"
                try:
					#send data to the GPR script
					syspf.get_TS(data['Humidity'])
                except:
					print "GPR failed"
				#post data to database
                r = requests.post("http://ee-ug1.ee.ic.ac.uk/actual_web2/house_measurement.php", data=data)
                print r.text
			#forward all messages with appropriate type
            elif data['type'] == 'light_request':
                for client in self.clients:
                    client.write_message(message)
            elif data['type'] == 'light_request_response':
                for client in self.clients:
                    client.write_message(message)
            elif data['type'] == 'light_control':
                for client in self.clients:
                    client.write_message(message)
            elif data['type'] == 'thermostat_control':
                for client in self.clients:
                    client.write_message(message)
            elif data['type'] == 'ping':
				print 'pong'
				#reply pong to a ping only to the client who sent it
				response = '{"type":"pong"}'
				self.write_message(response)
            elif data['type'] == 'thermostat_set':
                for client in self.clients:
                    client.write_message(message)
            elif data['type'] == 'thermostat_request':
                for client in self.clients:
                    client.write_message(message)
            elif data['type'] == 'thermostat_request_response':
                for client in self.clients:
                    client.write_message(message)
            else:
                print "Other data"
        #else:
        #    print 'message is not JSON'
 
    def on_close(self):
		#remove client from list
        self.clients.remove(self)
        print 'connection closed'
		#decrease user count
        global users
        users = users - 1
        print 'users : %d' % users
		#log it
        with open("tornado_ws.log", "a") as myfile:
            localtime = time.asctime( time.localtime(time.time()) )
            myfile.write('%s:connection closed : %d users \r\n' % (localtime,users))
	  
application = tornado.web.Application([
    (r"/", MainHandler),
    (r'/ws', WSHandler),
])

#set up to run as process
if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
	#listen to 8888
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
