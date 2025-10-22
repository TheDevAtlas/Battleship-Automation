import subprocess
import threading
import msvcrt  # Windows-only
import sys

SCRIPT_PATH = r"Battleship Auto-Play.py"  # change this to your script path

stop_flag = False

def key_listener():
    global stop_flag
    print("Press 'K' to stop.")
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode("utf-8").lower()
            if key == "k":
                print("Kill button pressed.")
                stop_flag = True
                break

def main():
    global stop_flag

    # Start background key listener
    listener = threading.Thread(target=key_listener, daemon=True)
    listener.start()

    while not stop_flag:
        # Launch the script
        process = subprocess.Popen([sys.executable, SCRIPT_PATH])

        # Wait until script finishes or kill pressed
        while process.poll() is None:
            if stop_flag:
                process.terminate()
                break

        if stop_flag:
            break

        print("Script finished. Restarting...")

if __name__ == "__main__":
    main()
