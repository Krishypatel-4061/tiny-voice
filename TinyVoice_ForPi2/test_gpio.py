"""
tools/test_gpio.py
==================
Verifies every actuator independently, BEFORE you connect the ML pipeline.

Golden rule of embedded debugging: never debug two subsystems at once. If you
only ever test the full system and the motor does not spin, you cannot tell
whether the model is wrong, the pin is wrong, or the wiring is wrong. This
script removes the model from the equation entirely.

USAGE (from the project root, on the Pi)
    python tools/test_gpio.py --all
    python tools/test_gpio.py --led
    python tools/test_gpio.py --motor
    python tools/test_gpio.py --servo
    python tools/test_gpio.py --interactive
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                   # noqa: E402
from gpio_control import ActuatorController     # noqa: E402


def test_led(act):
    print("\n[LED] Blinking 5 times on GPIO", config.PIN_LED)
    print("      Expect: visible blinking. If not -> check the 330 ohm resistor,")
    print("      LED polarity (long leg = anode = to the resistor), and GND.")
    for i in range(5):
        print("   ", act.toggle_led())
        time.sleep(0.5)
    if act.led_on:
        act.toggle_led()


def test_motor(act):
    print(f"\n[MOTOR] ON for 3s via L298N "
          f"(IN1={config.PIN_MOTOR_IN1}, IN2={config.PIN_MOTOR_IN2}, "
          f"ENA={config.PIN_MOTOR_ENA})")
    print("      Expect: motor spins. If it hums but does not turn ->")
    print("      raise MOTOR_SPEED in config.py or check the external 5-12V supply.")
    print("      If nothing at all -> verify the L298N has its own power AND")
    print("      that its GND is tied to the Pi's GND (common ground).")
    print("   ", act.toggle_motor())
    time.sleep(3)
    print("   ", act.toggle_motor())


def test_servo(act):
    print(f"\n[SERVO] Sweeping on GPIO {config.PIN_SERVO}")
    print("      Expect: smooth steps left and right, then return to centre.")
    print("      If it jitters violently -> it is being powered from the Pi's")
    print("      5V rail. Give it a separate supply with a common ground.")
    for _ in range(3):
        print("   ", act.move_servo(+config.SERVO_STEP_DEG))
        time.sleep(0.4)
    for _ in range(6):
        print("   ", act.move_servo(-config.SERVO_STEP_DEG))
        time.sleep(0.4)
    print("   ", act.stop_all())


def interactive(act):
    """Type command names by hand - simulates perfect speech recognition."""
    print("\nInteractive mode. Type a command and press Enter.")
    print("Valid:", ", ".join(config.COMMAND_WORDS), "| 'q' to quit")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("q", "quit", "exit"):
            break
        if cmd in config.COMMAND_WORDS:
            print("   ", act.handle_command(cmd))
            print("   ", act.status_line())
        else:
            print("    unknown command")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--led", action="store_true")
    p.add_argument("--motor", action="store_true")
    p.add_argument("--servo", action="store_true")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--simulate", action="store_true",
                   help="Force simulation mode (laptop testing)")
    a = p.parse_args()

    actuators = ActuatorController(simulate=True if a.simulate else None)
    try:
        if a.all or a.led:
            test_led(actuators)
        if a.all or a.motor:
            test_motor(actuators)
        if a.all or a.servo:
            test_servo(actuators)
        if a.interactive:
            interactive(actuators)
        if not any([a.all, a.led, a.motor, a.servo, a.interactive]):
            print("Nothing selected. Try --all or --interactive")
    finally:
        actuators.cleanup()
