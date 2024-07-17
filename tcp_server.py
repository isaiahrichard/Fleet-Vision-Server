import socket
import threading
import queue

# Used to store received data
data_queue = queue.Queue()

# TCP Server config for Boron MCU
TCP_IP = '192.168.1.18' #0.0.0.0 to listen on all available network interfaces
TCP_PORT = 5555
BUFFER_SIZE = 8096

def handle_client_connection(client_socket):
    try:
        data_buffer = bytearray()
        while True:
            data = client_socket.recv(BUFFER_SIZE)
            if not data:
                break
            data_buffer.extend(data) # append chunks of data to data buffer
            data_queue.put(data_buffer.decode("utf-8"))
            
            ack_msg = b'Data Received'
            client_socket.send(ack_msg)
            data_buffer = bytearray() # reset buff
    except Exception as e:
        print("Error: ", e)
    finally:
        client_socket.close()
        
def start_tcp_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((TCP_IP, TCP_PORT))
    server.listen(9)
    print("TCP server listening on {}:{}".format(TCP_IP, TCP_PORT))
    
    while True:
        client_socket, addr = server.accept()
        print('Accepted connection from: ', addr)
        client_handler = threading.Thread(target=handle_client_connection, args=(client_socket))
        client_handler.start()

if __name__ == "__main__":
    start_tcp_server()