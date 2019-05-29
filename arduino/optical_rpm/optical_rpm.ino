volatile long int REV;       //  VOLATILE DATA TYPE TO STORE REVOLUTIONS
 
unsigned long int rpm, maxRPM;  //  DEFINE RPM AND MAXIMUM RPM

unsigned long int DELAY = 1000;

unsigned long time;
unsigned long prevtime;
void setup() {
   Serial.begin(9600);   // GET VALUES USING SERIAL MONITOR
   pinMode(3, INPUT_PULLUP);
   attachInterrupt(digitalPinToInterrupt(3), RPMCount, FALLING);
   REV = 0;
}

void loop() {
  long currtime = millis();                 // GET CURRENT TIME
  
  long timedelta = currtime - prevtime;       //  CALCULATE IDLE TIME
  if (timedelta != 0) {
    float rpm = (REV * 1000) / timedelta ;
    Serial.print(rpm);
    Serial.print(",");
    Serial.print(REV);
    Serial.print(",");
    Serial.print(timedelta);
    Serial.println();
  }
  REV = 0;
  prevtime = currtime;
  
  delay(DELAY);
}

void RPMCount() {
 REV++;
}
