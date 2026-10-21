from gpiozero import Servo, LED
from RPLCD.i2c import CharLCD
from time import sleep

# Initialize Hardware
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)
led = LED(17)
servo = Servo(18)

try:
    # 1. Test LCD
    lcd.clear()
    lcd.write_string('Hardware Test\nRunning...')
    print("LCD Active")
    sleep(1)

    # 2. Test LED
    print("Blinking LED")
    for _ in range(3):
        led.on()
        sleep(0.5)
        led.off()
        sleep(0.5)

    # 3. Test Servo
    print("Sweeping Servo")
    servo.min()
    sleep(1)
    servo.max()
    sleep(1)
    servo.mid()
    
    lcd.clear()
    lcd.write_string('Test Complete!')
    print("Test Complete")

except KeyboardInterrupt:
    lcd.clear()
    led.off()
    servo.detach()