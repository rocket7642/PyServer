import socket
import sys
import threading
import FreeSimpleGUI as sg

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 25000  # Port to listen on (non-privileged ports are > 1023)

units = ""
eUnits = ""
eKUnits = ""

# All the stuff inside your window.
layout = [
    [sg.Multiline(size=(60, 10), key='-LOG-', autoscroll=True, disabled=True)],
    [sg.InputText()],
    [sg.Button('Ok'), sg.Button('Cancel')]
]

# Create the Window
window = sg.Window("Socket Reader", layout, finalize=True)

def receive_messages(conn):
    global units, eUnits, eKUnits
    
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break
            # Decode the received bytes into a string
            message = data.decode('utf-8')
            print(f"[{addr}] {message}")

            breakdown = message.split("\n")
            start = False

            if "KNOWN_ENEMY_UNITS" in message:
                for line in breakdown:
                    if line.strip() == "KNOWN_ENEMY_UNITS":
                        start = True
                        eKUnits = line + "\n"
                    elif start:
                        if line.strip() == "END":
                            start = False
                        else:
                            eKUnits += line + "\n"
            if "ENEMY_UNITS" in message:
                for line in breakdown:
                    if line.strip() == "ENEMY_UNITS":
                        start = True
                        eUnits = line + "\n"
                    elif start:
                        if line.strip() == "END":
                            start = False
                        else:
                            eUnits += line + "\n"
            if "FRIENDLY_UNITS" in message:
                for line in breakdown:
                    if line.strip() == "FRIENDLY_UNITS":
                        start = True
                        units = line + "\n"
                    elif start:
                        if line.strip() == "END":
                            start = False
                        else:
                            units += line + "\n"
            window.write_event_value('-SOCKET-', message)
        except:
            break
    print(f"[DISCONNECTED] {addr} disconnected.")
    conn.close()

# Create a socket object using the 'with' statement for automatic resource management
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.bind((HOST, PORT))  # Bind the socket to the address and port
        s.listen()  # Enable the server to accept connections
        print(f"Server listening on {HOST}:{PORT}...")

        # Accept an incoming connection. 'conn' is a new socket object for the client, 'addr' is the client's address
        conn, addr = s.accept()

        # sock.sendall(message.encode('utf-8'))
        server_thread = threading.Thread(target=receive_messages, args=(conn,), daemon=True)
        server_thread.start()
        print(f"[NEW CONNECTION] {addr} connected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


    while True:
        event, values = window.read()
        # if user closes window or clicks cancel
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break

        if event == 'Ok':
            user_input = values[0] + "\n"
            if user_input.strip() != "":
                # Run send in a separate thread to avoid blocking the GUI
                send_thread = threading.Thread(target=lambda: conn.sendall(user_input.encode('utf-8')), daemon=True)
                send_thread.start()
                print(f"User Input: {user_input}")

        # Handle incoming socket data
        if event == '-SOCKET-':
            data = values[event]
            window['-LOG-'].update(units + eUnits + eKUnits, append=False)

    window.close()




