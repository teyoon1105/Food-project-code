import socket

class Client:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False  # 연결 상태 저장

    def connect(self):
        if not self.connected:  # 이미 연결된 경우 다시 연결하지 않음
            try:
                self.client_socket.connect((self.server_ip, self.server_port))
                print(f"Connected to {self.server_ip}:{self.server_port}")
                self.connected = True
            except ConnectionRefusedError:
                print(f"Error: Could not connect to {self.server_ip}:{self.server_port}")
                return False
        return True

    def receive_data(self, buffer_size=1024):
        try:
            data = self.client_socket.recv(buffer_size)
            if not data:
                print("Connection closed by server.")
                self.connected = False  # 연결 끊김 표시
                return None
            return data.decode()
        except ConnectionResetError:
            print("Connection reset by server.")
            self.connected = False # 연결 끊김 표시
            return None
        
    def send_data(self, message):
        try:
            message_bytes = message.encode('utf-8')
            message_len = len(message_bytes).to_bytes(4, 'big') # 4바이트 길이
            print(f"message_len = {message_len}")
            
            
            sent_bytes1 = self.client_socket.sendall(message_len)
            sent_bytes2 = self.client_socket.sendall(message_bytes)
            
            print(f"sent_bytes1 = {sent_bytes1}")
            print(f"sent_bytes2 = {sent_bytes2}")
            
            return True
        except Exception as e:
            print(f"Error sending data: {e}")
            return False    
    

    # def send_data(self, message):
    #     try:
    #         message_bytes = message.encode('utf-8')
    #         message_len = len(message_bytes).to_bytes(4, 'big') # 4바이트 길이
    #         print(f"message_len = {message_len}")
            
            
    #         sent_bytes1 = self.client_socket.sendall(message_len)
    #         sent_bytes2 = self.client_socket.sendall(message_bytes)
            
    #         print(f"sent_bytes1 = {sent_bytes1}")
    #         print(f"sent_bytes2 = {sent_bytes2}")
            
    #         return True
    #     except Exception as e:
    #         print(f"Error sending data: {e}")



    def close(self):
        if self.connected: # 연결된 경우에만 소켓 닫기
            self.client_socket.close()
            self.connected = False
            print("Connection closed.")