#include "header.h"
#define PI 3.1415926

extern void swapaddress(char address[]);

extern char MR0address[8];
extern char MR2address[8];
extern bool lightStatus;

float PCurrent;
float PVoltage;
float PPhase;
float power;


void senddata()
{
/* function sends temperature, humidity, power and motion data back to the server at regular intervals */
    __disable_irq();
    powerconv();    //function to work out correct power measurement from calibration data
    
    sprintf(str, " {\"type\":\"house_measurement\" , \"Temperature\":%f, \"Humidity\":%f, \"power\":%f, \"Motion\":%f }", Temp, humidity, power, motionCount); //place in string
    //data sent after stored in string   
    wsoc.send(str);  
    __enable_irq();
}

/*
float cosine_func(float x){
    
    x = 1- 0.5*x*x+ (1/24)*x*x*x*x - (1/720)*x*x*x*x*x*x;
    return x;
}
*/

void powerconv()
{
/* function uses previously collected power data to calibrate the sensor correctly
    first working out the voltage, current and phase and then real power
    using known power equations */

    PVoltage = 0.3535*pVoltage - 2.6009;
    PCurrent = 0.0022*pCurrent - 0.0065;
    PPhase = (PI/1024)*pPhase;
    power = PVoltage*PCurrent*cos(PPhase);
}


void lightsignal()
{
/* function to turn on and off the lights, using the JSON string parsing to 
    find which light has been selected to turn off */

    int lightid = (int)v.get("light_ID").get<double>();
    //string parsing to get light id
    swapaddress(MR3address);
    //select correct xbee address
    
    /*switch function selects correct light and then using the "status" 
    variable from the JSON string sets light to on or off */
    switch( lightid ) {
        case 1:
            if(v.get("status").get<double>() == 1) {
                //call xbee and turn light on
                myxbee.SENDLIGHTONREQMOD();
                               
            } else {
                //call xbbe and turn light off
                myxbee.SENDLIGHTOFFREQMOD();
            }
            break;
        case 2:
            if(v.get("status").get<double>() == 1) {
                //call xbee and turn light on
            } else {
                //call xbbe and turn light off
            }
            break;
        default:
            break;
    }
}


void thermoupdate()
{
/* updates thermostat setting to correct variable from website */
    thermostat = v.get("setting").get<double>();
}


void recievedata()
{
/*  Function that is called when data is recieved from the websocket, 
    the string is then copied to a pointer type and parsed (using
    JSON functions) to determine the type and then calls other
    functions depending on the type
*/    
 
    char * json = (char*) malloc(strlen(recv)+1);
    strcpy(json, recv);
   
    string err = picojson::parse(v, json, json + strlen(json));
    //error function
    
    /*move the 'type' in JSON string into a string type which is then compared to 
    expected types from the server, after which actions are taken accordingly */
    char  type[200] ;
    strcpy(type,v.get("type").get<string>().c_str());
    printf("%s",type);
    if( strcmp(type, "light_request") == 0) {
        //request whether light is on or off
        sprintf(str, "{\"type\":\"light_request_response\" , \"light_ID\":1 , \"status\": %d }", lightStatus);
        wsoc.send(str);
    } else if( strcmp(type, "light_control") == 0) {
        //turn on or off light
        lightsignal();
    } else if( strcmp(type, "thermostat_request") == 0) {
        //request temperature setting of thermostat, which is returned, alond with house temperature
        sprintf(str, "{\"type\":\"thermostat_request_response\" , \"temperature\":%f , \"setting\": %f }", Temp, thermostat);
        wsoc.send(str);
    } else if( strcmp(type, "thermostat_control") == 0) {
        //thermostat setting has been updated from website
        thermoupdate();
    }

    return;

}
