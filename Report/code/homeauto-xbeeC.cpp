#include "mbed.h"
#include <vector>// alternate way of getting array length as standard wasnt working.
#include "MODSERIAL.h"
#include "xbee.h"


#define ATIDlength 50
#define ATISLENGTH 18
#define ATresponselength 9
#define MAXMESSAGELENGTH 100

DigitalOut myled(LED1);

extern MODSERIAL pc;
extern MODSERIAL xbee;


char ATIDplaceholder[ATIDlength];
//char BOSSaddress[8]= {0x00,0x13,0xA2,0x00,0x40,0x8B,0x71,0xDE};
//char MR1address[8]= {0x00,0x13,0xA2,0x00,0x40,0x8B,0x71,0x55};
char * MR1address;
char * test;
vector <char> comvec;
int hextoint(char MSB, char LSB)
{

    int total=0;

    total=(int)MSB*256+(int)LSB;

    return total;

}

void XBEE::swapaddress(char address[]){
//address pointed to becomes the one passed in argument.
MR1address=&address[0];

}

char * XBEE::findcurrentaddress(){

return MR1address;

}


char XBEE::checksum(char hexchar[])
{   //calculate checksume.
    int length;
    int calculatedchecksum;
    int total=0;

    length = (int)hexchar[1]+(int)hexchar[2];

    for(int i=3; i<3+length; i++) {//ignore delimiter and length bytes.
        total=total+(int)hexchar[i];
//pc.printf("%2x\n",hexchar[i]);
    }
    //pc.printf("%d\n",total);

    calculatedchecksum=0xFF-(total & 0xFF);
    //pc.printf("%2x\n",calculatedchecksum);


    return calculatedchecksum;
}


int XBEE::lastpacketpos(char packet[], int arraylength)//find last packet position in big reply string
{   //improvement -- detect all packet positions not just last one.

    int currentlastpos=0;

    //pc.printf("array length is...%df\n",arraylength);

    for(int i=0; i<arraylength; i++) { //static array, so valid method.

        if(packet[i]==0x7E)
            currentlastpos=i;
    }

    return currentlastpos;

}

void XBEE::readpacket(char packetstring[],int packetpos,int packetstringlength) 
{   

    if(packetstringlength != 0) {
        char snip[128];//makes a copy of current packet, a preparation for detecting multiple packets in returned string.
        enum packettype {ATcommandresponse,LocalATcommand,Modemstatus,Txreply,RemoteATcommandresponse,RemoteTxreply};
        enum packettype packetstatus;
        enum ATcommand {ATIS,ATID};//supported AT commands
        enum ATcommand command;
        int bytes =packetstring[packetpos+1]*256+packetstring[packetpos+2];
        //pc.printf("number of bytes in packet is %d\n",bytes);
        char status = packetstring[packetpos+3];//get frame specifier
        pc.printf("status is %2x\n",status);
        for(int j=0; j<bytes+2+1; j++) { //make a copy of current packet
            snip[j]=packetstring[packetpos+j];
        }

        char calculatedchecksum=checksum(snip);
        //pc.printf("calculated checksum is %2x\n",calculatedchecksum);
        //pc.printf("given checksum is %2x\n",packetstring[packetpos+bytes+2+1]);
        bool valid;


        if(calculatedchecksum==packetstring[packetpos+bytes+2+1]) { 
            switch (status) {//assign meaningful specifiers

                case 0x88:
                    packetstatus=ATcommandresponse;

                    pc.printf("AT command response packet detected\n");
                    break;
                case 0x8A:
                    packetstatus=Modemstatus;
                    break;
                case 0x08:
                    packetstatus=LocalATcommand;
                    pc.printf("LOCAL AT command response packet detected\n");
                    break;
                case 0x8B:
                    packetstatus=Txreply;
                    pc.printf("TX response packet detected\n");
                    if(packetstring[packetpos+2+5]==0x00)
                        pc.printf("packet was sent sucessfully\n");
                    else
                        pc.printf("Something went wrong\n");
                    break;
                case 0x97:
                    packetstatus=RemoteATcommandresponse;

                    pc.printf("AT command equivalent is %c%c\n",packetstring[packetpos+2+2+8+2+1],packetstring[packetpos+2+2+8+2+2]);//2/2/8/2===length offset/cmd type and frameID/64bit address/16 bit address
                    char pt1,pt2;
                    pt1=packetstring[packetpos+2+2+8+2+1];
                    pt2=packetstring[packetpos+2+2+8+2+2];
                    if(pt1=='I' && pt2=='S') {
                        command=ATIS;
                    }

                    break;
                case 0x90:
                    int messagestartpos =0;
                    int datapos=0;
                    packetstatus=RemoteTxreply;
                    pc.printf("REMOTE TX PACKET DETECTED");
                    messagestartpos=packetpos+2+2+8+2+1;
                    pc.printf("char %c%c%c \n", packetstring[messagestartpos],packetstring[messagestartpos+1],packetstring[messagestartpos+2]);
                    if(packetstring[messagestartpos]=='p' && packetstring[messagestartpos+1]=='w' && packetstring[messagestartpos+2]=='r') {//condition for power and not just a standard message

                        datapos=messagestartpos+3;
                        pc.printf("VCP is %2x%2x%2x%2x%2x%2x\n",packetstring[datapos],packetstring[datapos+1],packetstring[datapos+2],packetstring[datapos+3],packetstring[datapos+4],packetstring[datapos+5]);
                        //get raw values from power uC.
                        pVoltage=(int)packetstring[datapos]*256+(int)packetstring[datapos+1];
                        pCurrent=(int)packetstring[datapos+2]*256+(int)packetstring[datapos+3];
                        pPhase=(int)packetstring[datapos+4]*256+(int)packetstring[datapos+5];

                    } else {
                    //IMPROVEMENT -- implment message printer
                        pc.printf("standard message\n");

                    }

                    break;


            }

            if(command==ATIS) {
                pc.printf("SENSOR READING REQUESTED\n");
                pc.printf("%2x%2x\n",packetstring[packetpos+2+bytes-1],packetstring[packetpos+2+bytes]);
                int num=(packetstring[packetpos+2+bytes-1]*256)+packetstring[packetpos+2+bytes];
                float voltage=0;
                voltage=(float)num/1023*1.2;
                pc.printf("rawis %d\n",num);
                pc.printf("voltage is %fV\n",voltage);
                if(voltage<0.72&&packetstring[packetpos+2+2]==0x01/*gives us Frame ID*/){
                    pc.printf("movement detected\n");
                    motionCount++;
                    }
                else if(voltage >=0.72 && packetstring[packetpos+2+2]==0x01){//no motion detected if voltage is 0.8V
                    pc.printf("NADA\n");
                    }
                 else if(voltage < 0.5 && packetstring[packetpos+2+2]==0x02){
                 pc.printf("Light is off\n");//if frame ID is 2 then it must have been a light packet.
                 lightStatus=false;
                 
                 
                 }
                 else if(voltage>= 0.5 && packetstring[packetpos+2+2]==0x02){
                 pc.printf("Light is on \n");
                 lightStatus=true;
                 }

        }

        else {
            pc.printf("the packet is messed up in some way\n");
        }

    }

}
}

char * XBEE::createmessagepacket(char message[],char address[])
{

    static char packet[MAXMESSAGELENGTH +10];
    short totallength=0;
short length = 0;
    short setuplength = 17;//number of bytes required for address+options etc.
    while(message[length]!='\0') {
        length++;
    }

    pc.printf("length of input message data is %hd\n",length);

    totallength=length+setuplength;
    pc.printf("number of bytes in this packet is%hd\n",totallength);

    packet[0]=0x7E;
    packet[1]=(0xFF00 & (totallength-3));//-3 for frame delimiters and len MSB/LSB
    packet[2]=(0x00FF & (totallength-3));
    packet[3]=0x10;
    packet[4]=0x01;
    packet[5]=address[0];
    packet[6]=address[1];
    packet[7]=address[2];
    packet[8]=address[3];
    packet[9]=address[4];
    packet[10]=address[5];
    packet[11]=address[6];
    packet[12]=address[7];
    packet[13]=0xFF;
    packet[14]=0xFE;
    packet[15]=0x00;
    packet[16]=0x00;
    for(int i=0; i<length; i++) {
        packet[setuplength+i]=message[i];
        pc.printf("%c wass added to message\n",packet[setuplength+i]);
    }
    packet[setuplength+length]=checksum(packet);

    for(int i=0; i<totallength+1; i++) { //+1 to get chesum output.

        pc.printf("created packet byte %d is %2x\n",i+1,packet[i]);

    }

    return packet;

}

void XBEE::readreply(char reply[], int &replylength)//if/for implementation, doesnt work well unless reading something very specific.
{
    replylength =0;
    if(xbee.readable()) {

        for(int i=0; i<18; i++) {
            reply[i]=xbee.getc();
//pc.printf("reply byte %d iz %2x\n",replylength,reply[i]);

            replylength++;
        }

    }

}

void XBEE::readreply2(char reply[], int &replylength)//shows bytes read in.
{
    replylength =0;
    while(xbee.readable()) {


        reply[replylength]=xbee.getc();
        pc.printf("reply byte %d iz %2x\n",replylength,reply[replylength]);

        replylength++;



    }


    pc.printf("Length of reply is %d\n",replylength);

}


void XBEE::readreply3(char reply[], int &replylength)//no bytes shown, so completes a lot quicker as printf is bad for interrupts.
{
    replylength =0;
    while(xbee.readable()) {


        reply[replylength]=xbee.getc();

        replylength++;



    }




}



char * XBEE::createsensorpacket(char address[]) 
{

    static char packet[ATISLENGTH];
    short totallength=18;//number of bytes required for address+options etc.



    //pc.printf("number of bytes in this packet is%hd\n",totallength);

    packet[0]=0x7E;
    packet[1]=(0xFF00 & (totallength-3));//-3 for frame delimiters and len MSB/LSB
    packet[2]=(0x00FF & (totallength-3));
    packet[3]=0x17;
    packet[4]=0x01;
    packet[5]=address[0];
    packet[6]=address[1];
    packet[7]=address[2];
    packet[8]=address[3];
    packet[9]=address[4];
    packet[10]=address[5];
    packet[11]=address[6];
    packet[12]=address[7];
    packet[13]=0xFF;
    packet[14]=0xFE;
    packet[15]=0x00;
    packet[16]=0x49;//I
    packet[17]=0x53;//S
    packet[totallength]=checksum(packet);



    return packet;

}


char * XBEE::createlightStatuspacket(char address[]) //could be altered to void if needed
{
    

    static char packet[ATISLENGTH];
    short totallength=18;//number of bytes required for address+options etc.



    //pc.printf("number of bytes in this packet is%hd\n",totallength);

    packet[0]=0x7E;
    packet[1]=(0xFF00 & (totallength-3));//-3 for frame delimiters and len MSB/LSB
    packet[2]=(0x00FF & (totallength-3));
    packet[3]=0x17;
    packet[4]=0x02;
    packet[5]=address[0];
    packet[6]=address[1];
    packet[7]=address[2];
    packet[8]=address[3];
    packet[9]=address[4];
    packet[10]=address[5];
    packet[11]=address[6];
    packet[12]=address[7];
    packet[13]=0xFF;
    packet[14]=0xFE;
    packet[15]=0x00;
    packet[16]=0x49;//I
    packet[17]=0x53;//S
    packet[totallength]=checksum(packet);



    return packet;

}


char * XBEE::createlightonpacket(char address[]) //could be altered to void if needed
{

    static char packet[ATISLENGTH];
    short totallength=19;//number of bytes required for address+options etc.



    //pc.printf("number of bytes in this packet is%hd\n",totallength);

    packet[0]=0x7E;
    packet[1]=(0xFF00 & (totallength-3));//-3 for frame delimiters and len MSB/LSB
    packet[2]=(0x00FF & (totallength-3));
    packet[3]=0x17;
    packet[4]=0x01;
    packet[5]=address[0];
    packet[6]=address[1];
    packet[7]=address[2];
    packet[8]=address[3];
    packet[9]=address[4];
    packet[10]=address[5];
    packet[11]=address[6];
    packet[12]=address[7];
    packet[13]=0xFF;
    packet[14]=0xFE;
    packet[15]=0x02;
    packet[16]=0x44;
    packet[17]=0x30;
    packet[18]=0x05;//make output high
    packet[totallength]=checksum(packet);



    return packet;

}

char * XBEE::createlightoffpacket(char address[]) //could be altered to void if needed
{

    static char packet[ATISLENGTH];
    short totallength=19;//number of bytes required for address+options etc.



   // pc.printf("number of bytes in this packet is%hd\n",totallength);

    packet[0]=0x7E;
    packet[1]=(0xFF00 & (totallength-3));//-3 for frame delimiters and len MSB/LSB
    packet[2]=(0x00FF & (totallength-3));
    packet[3]=0x17;
    packet[4]=0x01;
    packet[5]=address[0];
    packet[6]=address[1];
    packet[7]=address[2];
    packet[8]=address[3];
    packet[9]=address[4];
    packet[10]=address[5];
    packet[11]=address[6];
    packet[12]=address[7];
    packet[13]=0xFF;
    packet[14]=0xFE;
    packet[15]=0x02;
    packet[16]=0x44;
    packet[17]=0x30;
    packet[18]=0x04;//make output low
    packet[totallength]=checksum(packet);



    return packet;

}

void XBEE::readaddress(char address[]){


pc.printf("address is %2x%2x%2x%2x%2x%2x%2x%2x \n",address[0],address[1],address[2],address[3],address[4],address[5],address[6],address[7]);

}//printf current address pointer



void XBEE::SENDMESSAGE()
{

    int length=0;
    int replylength=0;
    char cstr[MAXMESSAGELENGTH];
    char reply[MAXMESSAGELENGTH];
    char * msgptr;
    pc.printf("entermessage\n");
    pc.scanf("%s",cstr);
    while(cstr[length]!='\0') {//find length of input string
        length++;
    }



    pc.printf("length of input is %d\n",length);

    length=length+18;

    msgptr=createmessagepacket(cstr,MR1address);
    pc.printf("message packet created\n");

    if(xbee.writeable()) {//send packet down UART

        for(int i=0; i<length; i++) {

            xbee.putc(*(msgptr+i));
            pc.printf("contents of message packet byte %d is %2x\n", i,*(msgptr+i));

        }
    }
    readreply2(reply,replylength);
    readpacket(reply,lastpacketpos(reply,replylength),replylength);

    for(int i=0; i<replylength; i++) {
        pc.printf("reply is %2x and %d\n",reply[i],replylength);
    }


}


void XBEE::SENDSENSORREQ()
{
    __disable_irq();
    int length=19;
    int replylength=0;
    char reply[MAXMESSAGELENGTH];
    char * msgptr;
    char * temp=findcurrentaddress();

    
    
    

    msgptr=createsensorpacket(MR1address);
    pc.printf("SENSOR packet created\n");

    
    if(xbee.writeable()) {

        for(int i=0; i<length; i++) {
            //pc.printf("contents of message packet byte %d is %2x\n", i,*(msgptr+i));

            xbee.putc(*(msgptr+i));

        }
    }
    readreply2(reply,replylength);
    readpacket(reply,lastpacketpos(reply,replylength),replylength);



        swapaddress(temp);
        __enable_irq();
}

void XBEE::SENDMOTIONREQ(){
__disable_irq();
char * temp=findcurrentaddress();
swapaddress(MR0address);
SENDSENSORREQ();
swapaddress(temp);
__enable_irq();
}
void XBEE::SENDLIGHTONREQ()
{


    readaddress(MR1address);
    int length=20;
    int replylength=0;
    char reply[MAXMESSAGELENGTH];
    char * msgptr;



    msgptr=createlightonpacket(MR1address);
    //pc.printf("message packet created\n");


    if(xbee.writeable()) {

        for(int i=0; i<length; i++) {
//pc.printf("contents of message packet byte %d is %2x\n", i,*(msgptr+i));

            xbee.putc(*(msgptr+i));

        }
    }

}

void XBEE::SENDLIGHTOFFREQ()
{



    int length=20;
    int replylength=0;
    char reply[MAXMESSAGELENGTH];
    char * msgptr;



    msgptr=createlightoffpacket(MR1address);
    pc.printf("message packet created\n");


    if(xbee.writeable()) {

        for(int i=0; i<length; i++) {
//pc.printf("contents of message packet byte %d is %2x\n", i,*(msgptr+i));

            xbee.putc(*(msgptr+i));

        }
    }
readreply2(reply,replylength);
readpacket(reply,lastpacketpos(reply,replylength),replylength);




}


void xbeeATIDCHANGE()
{

    char PANID[2];
    short hex;

    char hexchar[10]= {0x7E,0x00,0x06,0x08,0x01,0x49,0x44,0x0E,0xE4,0x79};
    
    pc.printf("what do you want the PANID to be (in hex)?\n");
    pc.scanf("%x", &hex);//hex input
    pc.printf("%hd\n",hex);
    pc.printf("%4x\n",hex);
    PANID[1]=hex/256;//shift left bvy 8 bits so that can fit in char(0xFF00 is too big).
    PANID[0]=hex & 0x00FF;
    hexchar[7]=PANID[1];
    hexchar[8]=PANID[0];
    hexchar[9]=checksum(hexchar);

    

    pc.printf("BEGIN");
    for(int i=0; i<10; i++) {

        pc.printf("%2x\n",hexchar[i]);
    }


    pc.printf("END");

    if(xbee.writeable()) {

        for(int i=0; i<10; i++) {
            xbee.putc(hexchar[i]);

        }

    }




}






void XBEE::xbeeATID()
{

    char hexchar[8] = {0x7E,0x00,0x04,0x08,0x01,0x49,0x44,0x69};
    char temp;
    int length=0;
    checksum(hexchar);
    if(xbee.writeable()) {

        for(int i=0; i<=7; i++) {
            xbee.putc(hexchar[i]);

        }

    }



    for(int i=0; i<ATIDlength; i++) {
        if(xbee.readable()) {
            temp=xbee.getc();
            ATIDplaceholder[i]=temp;
            comvec.push_back(temp);
            pc.printf("%2x\n",temp);


        }

    }

    pc.printf("done\n");
    while(ATIDplaceholder[length]!='\0') {
        length++;
    }
    pc.printf("stringlength size is%d\n",length);
    pc.printf("vector size is .... %d\n",comvec.size());
    pc.printf("last packet pos starts at ... %d\n",lastpacketpos(ATIDplaceholder,comvec.size()));
    readpacket(ATIDplaceholder,lastpacketpos(ATIDplaceholder,comvec.size()),comvec.size());
    comvec.clear();


}

char * XBEE::createlightonpacketMOD(char address[],requestType request) //could be altered to void if needed
{

    static char packet[ATISLENGTH];
    short totallength=19;//number of bytes required for address+options etc.



   // pc.printf("number of bytes in this packet is%hd\n",totallength);

    packet[0]=0x7E;
    packet[1]=(0xFF00 & (totallength-3));//-3 for frame delimiters and len MSB/LSB
    packet[2]=(0x00FF & (totallength-3));
    packet[3]=0x17;
    packet[4]=0x00;
    packet[5]=address[0];
    packet[6]=address[1];
    packet[7]=address[2];
    packet[8]=address[3];
    packet[9]=address[4];
    packet[10]=address[5];
    packet[11]=address[6];
    packet[12]=address[7];
    packet[13]=0xFF;
    packet[14]=0xFE;
    packet[15]=0x02;
    packet[16]=0x44;//D
    packet[17]=0x30;//0
    packet[18]=0x05;//make output high
    packet[totallength]=checksum(packet);

    if(request==light){
    lightStatus=true;
}
    return packet;

}

char * XBEE::createlightoffpacketMOD    (char address[],requestType request) //could be altered to void if needed
{

    static char packet[ATISLENGTH];
    short totallength=19;//number of bytes required for address+options etc.



    //pc.printf("number of bytes in this packet is%hd\n",totallength);

    packet[0]=0x7E;
    packet[1]=(0xFF00 & (totallength-3));//-3 for frame delimiters and len MSB/LSB
    packet[2]=(0x00FF & (totallength-3));
    packet[3]=0x17;
    packet[4]=0x00;
    packet[5]=address[0];
    packet[6]=address[1];
    packet[7]=address[2];
    packet[8]=address[3];
    packet[9]=address[4];
    packet[10]=address[5];
    packet[11]=address[6];
    packet[12]=address[7];
    packet[13]=0xFF;
    packet[14]=0xFE;
    packet[15]=0x02;
    packet[16]=0x44;//D
    packet[17]=0x30;//0
    packet[18]=0x04;//make output low
    packet[totallength]=checksum(packet);

    if(request==light){
    lightStatus=false;
}
    return packet;

}

void XBEE::SENDLIGHTONREQMOD()
{   __disable_irq();



    int length=20;
    int replylength=0;
    char reply[MAXMESSAGELENGTH];
    char * msgptr;



    msgptr=createlightonpacketMOD(MR1address,light);
    pc.printf("light on packet created\n");


    if(xbee.writeable()) {

        for(int i=0; i<length; i++) {
//pc.printf("contents of message packet byte %d is %2x\n", i,*(msgptr+i));

            xbee.putc(*(msgptr+i));

        }
    }



    __enable_irq();

}

void XBEE::SENDLIGHTOFFREQMOD()
{   __disable_irq();

    

    int length=20;
    int replylength=0;
    char reply[MAXMESSAGELENGTH];
    char * msgptr;



    msgptr=createlightoffpacketMOD(MR1address,light);
    pc.printf("light off packet created\n");


    if(xbee.writeable()) {

        for(int i=0; i<length; i++) {
//pc.printf("contents of message packet byte %d is %2x\n", i,*(msgptr+i));
            xbee.putc(*(msgptr+i));
            
        }
    }


 __enable_irq();
}

void XBEE::SENDPOWERONREQMOD()
{   __disable_irq();



    int length=20;
    int replylength=0;
    char reply[MAXMESSAGELENGTH];
    char * msgptr;



    msgptr=createlightonpacketMOD(MR1address,power);
    pc.printf("light on packet created\n");


    if(xbee.writeable()) {

        for(int i=0; i<length; i++) {
//pc.printf("contents of message packet byte %d is %2x\n", i,*(msgptr+i));

            xbee.putc(*(msgptr+i));

        }
    }


    __enable_irq();

}

void XBEE::SENDPOWEROFFREQMOD()
{   __disable_irq();

    

    int length=20;
    int replylength=0;
    char reply[MAXMESSAGELENGTH];
    char * msgptr;



    msgptr=createlightoffpacketMOD(MR1address,power);
    pc.printf("light off packet created\n");


    if(xbee.writeable()) {

        for(int i=0; i<length; i++) {
//pc.printf("contents of message packet byte %d is %2x\n", i,*(msgptr+i));
            xbee.putc(*(msgptr+i));
            
        }
    }




 __enable_irq();
}

void XBEE::SENDPOWERREQ()
{   

__disable_irq();    
    char * temp=findcurrentaddress();
      /* for(int i=0;i<8;i++){
    pc.printf("address is %2X\n",MR1address[i]);
    }*/
    swapaddress(MR2address);//point to power XBEE
    /*for(int i=0;i<8;i++){
    pc.printf("address iss %2X\n",MR1address[i]);
    }*/
    char reply[MAXMESSAGELENGTH];
    int replylength=0;

    SENDPOWERONREQMOD();//send a digital on request with no reply for interrupt on rising edge for powrr uC
  

    wait(0.2);//wait for serial buffer to fill.
//pc.printf("INCOMING MESSAGE\n");
    readreply3(reply,replylength);//see what the uC replies with when it sends a message back
    readpacket(reply,lastpacketpos(reply,replylength),replylength);//extract raw data values from API frame.


    SENDPOWEROFFREQMOD();//set the interrupt pin on the uC to low ready for the next call.
    swapaddress(temp);
    
__enable_irq();
}

void XBEE::SENDLIGHTSTATUSREQ(){
__disable_irq();

    int length=19;
    int replylength=0;
    char reply[MAXMESSAGELENGTH];
    char * msgptr;
    char * temp=findcurrentaddress();
    /*for(int i=0;i<8;i++){
    pc.printf("address is %2X\n",MR1address[i]);
    }*/
    swapaddress(MR3address);//edit for #def power XBEE
    /*for(int i=0;i<8;i++){
    pc.printf("address iss %2X\n",MR1address[i]);
    }*/
    
    
    

    msgptr=createlightStatuspacket(MR1address);
    pc.printf("LIGHT STATUS packet created\n");


    if(xbee.writeable()) {

        for(int i=0; i<length; i++) {
            //pc.printf("contents of message packet byte %d is %2x\n", i,*(msgptr+i));

            xbee.putc(*(msgptr+i));

        }
    }
    readreply2(reply,replylength);
    readpacket(reply,lastpacketpos(reply,replylength),replylength);



        swapaddress(temp);
        __enable_irq();
        
 }
 

 







