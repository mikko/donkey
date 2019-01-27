#include <NewPing.h> //Iclud Arduino Pin Library
 
NewPing sonarLeft(TRIGGER_PIN_LEFT, ECHO_PIN_LEFT); 
NewPing sonarCenter(TRIGGER_PIN_CENTER, ECHO_PIN_CENTER); 
NewPing sonarRight(TRIGGER_PIN_RIGHT, ECHO_PIN_RIGHT); 

uint8_t prevLeft = 0;
uint8_t prevCenter = 0;
uint8_t prevRight = 0;

unsigned long prevTimeLeft = 0;
unsigned long prevTimeCenter = 0;
unsigned long prevTimeRight = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  } 
}
void loop() {
  uint8_t distanceLeft = sonarLeft.ping_cm(200);
  unsigned long timeLeft = millis();
  uint8_t distanceCenter = sonarCenter.ping_cm(200);
  unsigned long timeCenter = millis();
  uint8_t distanceRight = sonarRight.ping_cm(200);
  unsigned long timeRight = millis();

  Serial.print(distanceLeft);
  Serial.print(" ");
  Serial.print(distanceCenter);
  Serial.print(" ");
  Serial.print(distanceRight);
  Serial.print(" ");
  // Distance change from previous update
  float deltaDistance = prevCenter - distanceCenter;
  float deltaTime = timeCenter - prevTimeCenter;
  
  float impactEstimate = distanceCenter / (deltaDistance / deltaTime);
  // Serial.print(impactEstimate);
  if (impactEstimate > 0 && impactEstimate < 3000) {
    Serial.print(impactEstimate / 1000);
  } else {
    Serial.print(-1);
  }
  Serial.println(" ");
  prevTimeLeft = timeLeft;
  prevTimeCenter = timeCenter;
  prevTimeRight = timeRight;
  prevLeft = distanceLeft;
  prevCenter = distanceCenter;
  prevRight = distanceRight;
  delay(50);
}
