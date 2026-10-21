"""
gpio_control.py
===============
Everything that touches physical hardware lives here, and nowhere else.

This separation matters for your team: you can develop and test the whole ML
pipeline on a laptop with `simulate=True`, and the day the hardware is ready
you flip one flag. No ML code has to change.

LIBRARY CHOICE: gpiozero
------------------------
gpiozero is the officially supported GPIO library on Raspberry Pi OS. It is
preferred over RPi.GPIO because it works on modern Raspberry Pi OS (Bookworm)
via the lgpio backend, has proper device classes, and cleans up pins
automatically on exit.

ELECTRICAL SAFETY - READ THIS BEFORE WIRING TOMORROW
----------------------------------------------------
* A Raspberry Pi GPIO pin is 3.3 V and can source about 16 mA. The whole chip
  should stay under ~50 mA total.
* NEVER connect a DC motor directly to a GPIO pin. The stall current and the
  inductive kickback when it stops will destroy the Pi. Always use a driver
  (L298N / L293D / a MOSFET with a flyback diode).
* NEVER feed 5 V into a GPIO input. The Pi is NOT 5 V tolerant.
* Power the motor and servo from a SEPARATE 5 V supply, not the Pi's 5 V rail.
  A servo's startup current spike will brown-out the Pi and reboot it
  mid-demo - a very common and very visible failure.
* The Pi's ground and the external supply's ground MUST be connected together
  (common ground), or none of the signals have a reference and nothing works.
"""

import time

import config

# Try to import gpiozero. On a laptop this fails, which is fine - we fall back
# to simulation mode and print what WOULD have happened.
try:
    from gpiozero import LED, PWMOutputDevice, DigitalOutputDevice, AngularServo
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False


class ActuatorController:
    """
    Maps recognised commands to physical actions.

    Command mapping (from the project proposal):
        "light" -> toggle the LED
        "fan"   -> toggle the DC motor
        "up"    -> servo angle += SERVO_STEP_DEG
        "down"  -> servo angle -= SERVO_STEP_DEG
        "stop"  -> everything off, servo back to centre
    """

    def __init__(self, simulate=None):
        # If the caller did not force a mode, decide automatically:
        # real hardware if gpiozero imported, simulation otherwise.
        self.simulate = (not GPIO_AVAILABLE) if simulate is None else simulate

        # Software state, kept regardless of mode so the logic is testable.
        self.led_on = False
        self.motor_on = False
        self.servo_angle = 0

        if self.simulate:
            print("[GPIO] SIMULATION MODE - no pins will be driven.")
            return

        print("[GPIO] Initialising hardware...")

        # --- LED --------------------------------------------------------
        # Wire: GPIO17 -> 330 ohm resistor -> LED anode (long leg);
        #       LED cathode (short leg) -> GND.
        # The resistor is not optional: without it the LED draws too much
        # current and can damage the pin.
        self.led = LED(config.PIN_LED)

        # --- DC motor via L298N ------------------------------------------
        # IN1/IN2 set the direction (one HIGH, one LOW = spin one way).
        # ENA is a PWM pin controlling speed via duty cycle.
        self.motor_in1 = DigitalOutputDevice(config.PIN_MOTOR_IN1)
        self.motor_in2 = DigitalOutputDevice(config.PIN_MOTOR_IN2)
        self.motor_ena = PWMOutputDevice(config.PIN_MOTOR_ENA, frequency=1000)

        # --- Servo --------------------------------------------------------
        # min/max_pulse_width are the standard hobby-servo timings: a 0.5 ms
        # pulse = one extreme, 2.5 ms = the other, repeated every 20 ms.
        # If your servo buzzes or does not reach the full range, adjust these
        # two numbers - they vary between servo brands.
        self.servo = AngularServo(
            config.PIN_SERVO,
            min_angle=config.SERVO_MIN_DEG,
            max_angle=config.SERVO_MAX_DEG,
            min_pulse_width=0.5 / 1000,
            max_pulse_width=2.5 / 1000,
        )
        self.servo.angle = 0
        time.sleep(0.4)
        # Detaching the servo after it has moved stops the constant PWM jitter
        # and the buzzing noise. It will re-attach on the next angle assignment.
        self.servo.detach()

        print("[GPIO] Hardware ready.")

    # ======================================================================
    # Individual actions
    # ======================================================================
    def toggle_led(self):
        self.led_on = not self.led_on
        if not self.simulate:
            self.led.on() if self.led_on else self.led.off()
        return f"LED {'ON' if self.led_on else 'OFF'}"

    def toggle_motor(self):
        self.motor_on = not self.motor_on
        if not self.simulate:
            if self.motor_on:
                # Direction: IN1 high, IN2 low.
                self.motor_in1.on()
                self.motor_in2.off()
                self.motor_ena.value = config.MOTOR_SPEED
            else:
                self.motor_ena.value = 0
                self.motor_in1.off()
                self.motor_in2.off()
        return f"MOTOR {'ON' if self.motor_on else 'OFF'} " \
               f"(speed {int(config.MOTOR_SPEED * 100)}%)"

    def move_servo(self, delta_deg):
        # Clamp so we never command the servo past its mechanical limits -
        # doing so makes it stall, heat up and draw large current.
        new_angle = self.servo_angle + delta_deg
        new_angle = max(config.SERVO_MIN_DEG, min(config.SERVO_MAX_DEG, new_angle))

        if new_angle == self.servo_angle:
            return f"SERVO already at limit ({self.servo_angle} deg)"

        self.servo_angle = new_angle
        if not self.simulate:
            self.servo.angle = self.servo_angle
            time.sleep(0.35)        # give it time to physically travel
            self.servo.detach()     # then stop driving it (kills the buzz)
        return f"SERVO -> {self.servo_angle} deg"

    def stop_all(self):
        self.led_on = False
        self.motor_on = False
        self.servo_angle = 0
        if not self.simulate:
            self.led.off()
            self.motor_ena.value = 0
            self.motor_in1.off()
            self.motor_in2.off()
            self.servo.angle = 0
            time.sleep(0.35)
            self.servo.detach()
        return "ALL STOPPED (LED off, motor off, servo centred)"

    # ======================================================================
    def handle_command(self, command):
        """
        The single entry point used by the main loop.
        Returns a human-readable description of what happened, for the console
        and the optional OLED.
        """
        if command == "light":
            return self.toggle_led()
        if command == "fan":
            return self.toggle_motor()
        if command == "up":
            return self.move_servo(+config.SERVO_STEP_DEG)
        if command == "down":
            return self.move_servo(-config.SERVO_STEP_DEG)
        if command == "stop":
            return self.stop_all()
        return f"(no action mapped for '{command}')"

    def status_line(self):
        """Compact one-line state summary."""
        return (f"LED={'ON ' if self.led_on else 'OFF'}  "
                f"MOTOR={'ON ' if self.motor_on else 'OFF'}  "
                f"SERVO={self.servo_angle:+4d}deg")

    def cleanup(self):
        """Release every pin. Always call this on exit."""
        if self.simulate:
            return
        try:
            self.stop_all()
            self.led.close()
            self.motor_in1.close()
            self.motor_in2.close()
            self.motor_ena.close()
            self.servo.close()
            print("[GPIO] Pins released.")
        except Exception as e:
            print(f"[GPIO] Cleanup warning: {e}")
