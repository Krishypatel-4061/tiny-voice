from flask import Flask, render_template, request, redirect
from gpiozero import Servo, LED
from RPLCD.i2c import CharLCD
from time import sleep

app = Flask(__name__)

# 1. Safely Initialize Hardware
# Ensure the I2C address is correct (0x27 is standard, some are 0x3F)
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)
led = LED(17)
servo = Servo(18)

# Clear the LCD on boot
lcd.clear()
lcd.write_string('System Ready')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/command', methods=['POST'])
def command():
    cmd = request.form.get('action')
    
    # Safely clear the LCD and print the new command
    lcd.clear()
    lcd.write_string(f'Cmd: {cmd}')
    
    # Trigger the hardware based on the web button clicked
    if cmd == 'LIGHT ON':
        led.on()
    elif cmd == 'LIGHT OFF':
        led.off()
    elif cmd == 'UP':
        servo.max()  # Moves servo to +90 degrees
    elif cmd == 'DOWN':
        servo.min()  # Moves servo to -90 degrees
        
    return redirect('/') # Reload the page instantly

if __name__ == '__main__':
    # host='0.0.0.0' allows your laptop to access the Pi's server
    app.run(host='0.0.0.0', port=5000)