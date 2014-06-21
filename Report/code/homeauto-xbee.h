#ifndef XBEE_H
#define XBEE_H
#include "mbed.h"
#include <vector>
#include "MODSERIAL.h"

//external globals declared in main loop which allow measurements to be passed from this library to the main loop.

extern int pVoltage;
extern int pCurrent;
extern int pPhase;
extern int motionCount;

//addresses of the XBees

extern char MR0address[8];//motion
extern char MR2address[8];//power
extern char MR3address[8];//light control
extern bool lightStatus;//current light status

class XBEE{

char checksum(char hexchar[]);//calculates checksum of given packet
enum requestType{light,power};//enumeration for differentiating between light control analogue readings and power readings
void readpacket(char packetstring[],int packetpos,int packetstringlength);//decode packet information.
char * createmessagepacket(char message[],char address[]);//create API frame for sending ASCII messages
void readreply(char reply[], int &replylength);
void readreply2(char reply[], int &replylength);//read the reply with all bytes shown
void readreply3(char reply[],int &replylength);//read the reply with no prints (good for interrupts).
char * createsensorpacket(char address[]);
char * createlightStatuspacket(char address[]);
char * createlightonpacket(char address[]);
char * createlightoffpacket(char address[]);
char * createlightoffpacketMOD(char address[],requestType request);//MODIFIED packet so that there is no reply from the receiver XBEE so that it doesn't confuscate the ASCII string sent from power XBEE
char * createlightonpacketMOD(char address[],requestType request);
void readaddress(char address[]);
char * findcurrentaddress();//return pointer to array holding current address.







public:

void SENDSENSORREQ();//normal analogue reading on AD1
void SENDMOTIONREQ();//send packet and decode motion sensor reply
void SENDLIGHTONREQ();//turn light on using DO
void SENDLIGHTOFFREQ();//turn light off using
void SENDMESSAGE();
void xbeeATID();//find PAN ID
void xbeeATIDCHANGE();//change PAN ID
void SENDLIGHTONREQMOD();//no reply for power uC
void SENDLIGHTOFFREQMOD();//no reply for power uC
void SENDPOWERONREQMOD();//no reply for power uC
void SENDPOWEROFFREQMOD();//no reply for power uC
void SENDPOWERREQ();
void SENDLIGHTSTATUSREQ();//find status of light connected to XBee which is currently pointed to
void swapaddress(char address[]);
int lastpacketpos(char packet[], int arraylength);







};

#endif