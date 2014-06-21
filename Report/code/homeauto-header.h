#ifndef header_h
#define header_h
#include "Websocket.h"

#include "picojson.h"
#include "mbed.h"
#include "xbee.h"






extern Websocket wsoc; //websocket address
extern picojson::value v;

extern char str[512];
extern char recv[512];
extern float humidity;
extern float Temp;
extern float power;

extern float thermostat;
extern XBEE myxbee;

void lightsignal();
void senddata();
void thermoupdate();
void recievedata();
void powerconv();

#endif