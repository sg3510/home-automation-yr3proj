#include "mbed.h"
#include "TMP175.h"
#include "WiflyInterface.h"
#include "HTTPClient.h"
#include "Websocket.h"
#include "SDFileSystem.h"
#include "picojson.h"
#include "header.h"
#include "xbee.h"
#include "MODSERIAL.h"
#define ATIClength 23 // 23 gives best results 


AnalogIn ain(p18);          // Humidity pin
Serial wifi(p13,p14);       // Wifly module pins
DigitalOut rst1(p26);       // Temperature reset pin
DigitalOut WiFlyreset(p30);     //direct reset signal into WiFly for errorhandler
WiflyInterface wifly(p13, p14,p19,p20,"TP-LINK_804327", "C1804327", WPA); //wifly module set-up
SDFileSystem sd(p5, p6, p7, p8, "sd"); // SD card set up

//Tickers set for recurring functions
Ticker sendPower; 
Ticker measureMotion;
Ticker lightstatus;
Ticker dataSend;

TMP175 mySensor(p28,p27);   // Temperature set up
Websocket wsoc("ws://ee-ug1.ee.ic.ac.uk:8888/ws"); //websocket address
picojson::value v;  //JSON parsing variable
XBEE myxbee;    //Xbee function
MODSERIAL pc(USBTX, USBRX); //Debugging using serial terminal
MODSERIAL xbee(p9,p10);     //Modserial allows increased buffer size for xbee
DigitalOut led(LED1);
DigitalOut flash(LED4);

//initialise variables
char response[ATIClength];
char orderedresponse[ATIClength];
char str[512];
char recv[512];
float humidity;
float Temp;
float thermostat = 15;
int pCurrent=0;
int pPhase=0;
int pVoltage=0;
int motionCount =0;
int bufferpos= 0;
int a=0;
//64 bit addresses of XBEEs
char MR0address[8]= {0x00,0x13,0xA2,0x00,0x40,0x8B,0x71,0x55};//motion sensor XBEE 
char MR2address[8]= {0x00,0x13,0xA2,0x00,0x40,0x8B,0x70,0x79};//power monitoring XBEE
char MR3address[8]= {0x00,0x13,0xA2,0x00,0x40,0x8B,0x71,0x45};//Light control XBEE
//extern char MR1address[8];
extern char * test;
extern char * MR1address;//pointer to array which holds the currently selected address.
bool lightStatus=false;


void swapaddress(char address[])
{   
/* used to swap xbee address as required for communication with the
    different xbees */

    MR1address=&address[0];

}

void readATICresponseISR(MODSERIAL_IRQ_INFO *q)
{

    MODSERIAL * serial= q->serial;


    if(serial->readable())
        response[bufferpos]=serial->rxGetLastChar();//read in char but dont delete from circular buffer

    bufferpos++;
    pc.printf("IRQ REPLAY %d IS %2X LONG\n",bufferpos, response[bufferpos]);
    if(bufferpos == ATIClength) {
        bufferpos = 0;
    }
    if(serial->rxBufferFull()==true) {
        serial->rxBufferFlush();//flush the buffer once it is full.
        if(serial->rxBufferEmpty()==true)
            pc.printf("BUFFER FLUSHED\n");
        bufferpos=0;
    }

}

void readATICresponseISR2()
{

    int pos;
    if(xbee.readable()) {
        response[bufferpos]=xbee.getc();//clear bytes from FIFO
    }

    bufferpos++;
    pc.printf(" %d IS %2X LONG\n",bufferpos, response[bufferpos]);

    if(bufferpos == ATIClength-1) {
        bufferpos = 0;
        pos=myxbee.lastpacketpos(response,ATIClength);


        pc.printf("packet starts here at byte %d pos\n the packet delimiter is %2X \n",pos+1,response[pos]);
        pc.printf("DIGITAL HEX VARS ARE %2X and %2X\n",response[ATIClength-4],response[ATIClength-3]);//extract the digital values from sent packet

    }

}

void lightCheck()
{
    //function to call xbee light check function and to send the status to the sever

    __disable_irq();
    myxbee.SENDLIGHTSTATUSREQ();
    //sprintf(str, "{\"type\":\"light_request_response\" , \"light_ID\":1 , \"status\": %d }", lightStatus);
    //wsoc.send(str);
    __enable_irq();
}

void pingpong()
{
/*  ping pong function used to detect whether websockets are still running
    and then reset and reboot wifly module then reconnect to websocket */
    
    
    __disable_irq();
    
    pc.printf("is connected %d\n",wifly.is_connected());
    wsoc.send("{\"type\":\"ping\"}");   //send ping tp sever
    wait_ms(100);
    //if server responds then websockets are working so return//
    if(wsoc.read(recv)) {
        pc.printf("%s\n",recv);
        //if(strcmp("{\"type\":\"pong\"}",recv) == 0)
        __enable_irq();
        return;
    }
    //otherwise reset and reboot wifly module
    WiFlyreset=0;
    pc.printf("RESETTING WIFLY\n");
    wait_ms(200);//make an active low pulse
    WiFlyreset=1;
    wait_ms(100);//give time to reset
    wifly.send("reboot",6); //send reboot cmd to wifly module
    wait_ms(200);
    wifly.init(); //Use DHCP
    wait(0.5);
    wifly.connect();  //connect to wifi
    wait(0.5);
    
    //websocket reconnect

    if(wsoc.connect() == 1) {
        pc.printf("This was a truimph!");
    } else {
        pc.printf("failed!");
        // pingpong();
    }
    __enable_irq();
    return;

}


int main()
{   
    //set baud rate
    pc.baud(115200);
    xbee.baud(115200);

    //wifly reset at start
    __disable_irq();
    WiFlyreset=1;
    wait(2);
    WiFlyreset=0;
    wait(2);
    WiFlyreset=1;
    
    //set xbee addresses
    MR1address=&MR2address[0];
    
    
    pc.printf("Test Wifly!\r\n");
    

    mySensor.vSetConfigurationTMP175(SHUTDOWN_MODE_OFF|COMPARATOR_MODE|POLARITY_0|FAULT_QUEUE_6|RESOLUTION_9,0x48); // Temperature set-up
    wait_ms(400);
    wifly.reboot(); //reboot wifly
    wait(1.0);
    wifly.init(); //Use DHCP
    wait(1.0);
    wifly.connect();  //connect to wifi
    wait(1.0);
    wsoc.connect(); //connect to websocket
    __enable_irq();
    
    
    sendPower.attach(&myxbee,&XBEE::SENDPOWERREQ, 5);
    //ticker polling power sensor for data
    wait_ms(1000);
    
    measureMotion.attach(&myxbee,&XBEE::SENDMOTIONREQ,3.0);
    //ticker polling motion sensor
    wait_ms(1000);
    
    lightstatus.attach(&lightCheck,3);
    //ticker polling whether the light is on or off
    
    dataSend.attach(&senddata,17);
    //send power, humidity, temperature and motion data to server

    while (1) {


        //***************************Temp sensor*****************************************//

        Temp=mySensor; //get temperature in variable 'temp'

        //*************************Humidity Sensor***************************************//
                
        humidity = ain.read();
        humidity = (((humidity*5)/3.3)-0.1515)/0.00636;
        humidity = humidity/(1.0546-(0.00216*Temp));
                      
        

        //**********************recieve section****************************************//

        recv[0] = '\0';         //set recieve string to 0
        wait_ms(1500);
        
        __disable_irq();
        if(wsoc.read(recv)) {   //if communication recieve pass data to recieve function

            pc.printf("%s\r\n", recv);
            recievedata();

        }
        else {
            //otherwise check websocket is still running
            pingpong();
        }
        __enable_irq();
        
        
        //end of continious while loop
    }

}
