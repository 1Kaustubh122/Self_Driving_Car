import os
import sys
import time
import socket
import subprocess

CARLA_IP = "192.168.0.2"
CARLA_PORTS = [2000, 2003, 2006, 2009]

def is_port_open(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect((ip, port))
        return True
    except Exception:
        return False
    finally:
        s.close()

def check_carla_servers():
    for port in CARLA_PORTS:
        if not is_port_open(CARLA_IP, port):
            print(f"[Monitor] Port {port} down!")
            return False
    return True

def start_training():
    return subprocess.Popen([sys.executable, "train.py"])

def kill_training(proc):
    if proc is not None:
        proc.kill()

def main():
    training_proc = None

    while True:
        if not check_carla_servers():
            print("[Monitor] Carla DOWN! Stopping training...")
            kill_training(training_proc)

            print("[Monitor] Waiting for Carla to come back up...")
            while not check_carla_servers():
                time.sleep(5)

            print("[Monitor] Carla back! Restarting training...")
            training_proc = start_training()
        else:
            if training_proc is None or training_proc.poll() is not None:
                print("[Monitor] Training process not running, starting...")
                training_proc = start_training()

        time.sleep(10)

if __name__ == "__main__":
    main()
