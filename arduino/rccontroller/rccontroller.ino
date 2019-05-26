// Timer2 installation instructions can be found here:
// https://github.com/ElectricRCAircraftGuy/eRCaGuy_TimerCounter

#include <eRCaGuy_Timer2_Counter.h>

#define fastDigitalRead(p_inputRegister, bitMask) ((*p_inputRegister & bitMask) ? HIGH : LOW)

// Pin configuration
// Supported D0-D13 & A0-A5
const byte steering_pin = 2;
const byte throttle_pin = 3;
const byte mode_pin = 4;

byte steering_input_pin_bitmask;
volatile byte* p_steering_input_pin_register;

byte throttle_input_pin_bitmask;
volatile byte* p_throttle_input_pin_register;

byte mode_input_pin_bitmask;
volatile byte* p_mode_input_pin_register;

volatile bool output_data = false;

// In 0.5 microseconds
volatile unsigned long steering_pulse_counts = 0;
volatile unsigned long throttle_pulse_counts = 0;
volatile unsigned long mode_pulse_counts = 0;

void setup() 
{
  pinMode(steering_pin, INPUT_PULLUP);
  pinMode(throttle_pin, INPUT_PULLUP);
  pinMode(mode_pin, INPUT_PULLUP);

  timer2.setup();

  steering_input_pin_bitmask = digitalPinToBitMask(steering_pin);
  p_steering_input_pin_register = portInputRegister(digitalPinToPort(steering_pin));

  throttle_input_pin_bitmask = digitalPinToBitMask(throttle_pin);
  p_throttle_input_pin_register = portInputRegister(digitalPinToPort(throttle_pin));

  mode_input_pin_bitmask = digitalPinToBitMask(mode_pin);
  p_mode_input_pin_register = portInputRegister(digitalPinToPort(mode_pin));
  
  configurePinChangeInterrupts();
  
  Serial.begin(115200);
}

void loop() 
{
  static float steeringPulseTime = 0;
  static float throttlePulseTime = 0;
  static float modePulseTime = 0;
  
  if (output_data)
  {
    unsigned long steeringPulseCountsCopy;
    unsigned long throttlePulseCountsCopy;
    unsigned long modePulseCountsCopy;
    
    // Turn off interrupts while reading the data
    noInterrupts();

    steeringPulseCountsCopy = steering_pulse_counts;
    throttlePulseCountsCopy = throttle_pulse_counts;
    modePulseCountsCopy = mode_pulse_counts;
    output_data = false;
    
    interrupts();

    Serial.print(steeringPulseCountsCopy);
    Serial.print(",");
    Serial.print(throttlePulseCountsCopy);
    Serial.print(",");
    Serial.println(modePulseCountsCopy);
  }
}

void pinChangeIntISR()
{
  static boolean pin_state_steering_new = LOW;
  static boolean pin_state_steering_old = LOW;
  static boolean pin_state_throttle_new = LOW;
  static boolean pin_state_throttle_old = LOW;
  static boolean pin_state_mode_new = LOW;
  static boolean pin_state_mode_old = LOW;
  
  static unsigned long timer_start_steering = 0;
  static unsigned long timer_start_steering_old = 0;
  static unsigned long timer_start_throttle = 0;
  static unsigned long timer_start_throttle_old = 0;
  static unsigned long timer_start_mode = 0;
  static unsigned long timer_start_mode_old = 0;
  
  pin_state_steering_new = fastDigitalRead(p_steering_input_pin_register, steering_input_pin_bitmask);
  pin_state_throttle_new = fastDigitalRead(p_throttle_input_pin_register, throttle_input_pin_bitmask);
  pin_state_mode_new = fastDigitalRead(p_mode_input_pin_register, mode_input_pin_bitmask);

  if (pin_state_steering_old != pin_state_steering_new)
  {
    pin_state_steering_old = pin_state_steering_new;
    if (pin_state_steering_new == HIGH)
    {
      timer_start_steering = timer2.get_count();
      timer_start_steering_old = timer_start_steering;
    } else
    {
      unsigned long timer_end = timer2.get_count();
      steering_pulse_counts = timer_end - timer_start_steering;
      output_data = true;
    }
  }
  
  else if (pin_state_throttle_old != pin_state_throttle_new)
  {
    pin_state_throttle_old = pin_state_throttle_new;
    if (pin_state_throttle_new == HIGH)
    {
      timer_start_throttle = timer2.get_count();
      timer_start_throttle_old = timer_start_throttle;
    } else
    {
      unsigned long timer_end = timer2.get_count();
      throttle_pulse_counts = timer_end - timer_start_throttle;
      output_data = true;
    }
  }

  else if (pin_state_mode_old != pin_state_mode_new)
  {
    pin_state_mode_old = pin_state_mode_new;
    if (pin_state_mode_new == HIGH)
    {
      timer_start_mode = timer2.get_count();
      timer_start_mode_old = timer_start_mode;
    } else
    {
      unsigned long timer_end = timer2.get_count();
      mode_pulse_counts = timer_end - timer_start_mode;
      output_data = true;
    }
  }
}

// Pins D8 to D13
ISR(PCINT0_vect)
{
  pinChangeIntISR();
}

// Pins A0 to A5
ISR(PCINT1_vect)
{
  pinChangeIntISR();
}

// Pins D0 to D7
ISR(PCINT2_vect)
{
  pinChangeIntISR();
}

void configurePinChangeInterrupts()
{
  // Enable pin changes interrupts by setting pin change mask register & pin change interrupt 
  // control registers on for related pins
  volatile byte* p_steering_PCMSK = (volatile byte*)digitalPinToPCMSK(steering_pin);
  *p_steering_PCMSK |= _BV(digitalPinToPCMSKbit(steering_pin));
  volatile byte* p_steering_PCICR = (volatile byte*)digitalPinToPCICR(steering_pin);
  *p_steering_PCICR |= _BV(digitalPinToPCICRbit(steering_pin));

  volatile byte* p_throttle_PCMSK = (volatile byte*)digitalPinToPCMSK(throttle_pin);
  *p_throttle_PCMSK |= _BV(digitalPinToPCMSKbit(throttle_pin));
  volatile byte* p_throttle_PCICR = (volatile byte*)digitalPinToPCICR(throttle_pin);
  *p_throttle_PCICR |= _BV(digitalPinToPCICRbit(throttle_pin));

  volatile byte* p_mode_PCMSK = (volatile byte*)digitalPinToPCMSK(mode_pin);
  *p_mode_PCMSK |= _BV(digitalPinToPCMSKbit(mode_pin));
  volatile byte* p_mode_PCICR = (volatile byte*)digitalPinToPCICR(mode_pin);
  *p_mode_PCICR |= _BV(digitalPinToPCICRbit(mode_pin));
}
