from server_client import Client
import time

class ServerTest:
    
    def __init__(self):
        # 서버는 항상 켜두고 -> 계속 라즈베리파이랑 통신하게 두고
        self.client = Client('172.31.99.10', 5000) # 라즈베리파이 IP 주소 사용
        # self.client = Client('192.168.191.100', 5000) # 라즈베리파이 IP 주소 사용
        self.client.connect()
    
    def get_weight(self):
        try:
            if self.client.connect():
                weight = self.client.receive_data()
                if weight is None:
                    print("Failed to receive data. Closing connection.")
                return round(float(weight), 2)
        except Exception as e:
            return print(f"Error : {e}")
    
    # 카메라에서 detect된 obj의 무게를 가져옴
    def get_weight_food(self, obj):
        try:
            if self.client.connect():
                # while True:
                # 송신
                self.client.send_data(obj)
                print(f"send obj = {obj}")
                # time.sleep(5)
                    # else:
                    # print("Failed to send message. Closing connection.")
                        # break  # 연결 종료
                    
                # 수신
                data = self.client.receive_data()
                if data is None:
                    print("Failed to receive data. Closing connection.")
                    # break  # 연결 종료
                print(f"{obj} = {data}")
                return obj, data
            
        except Exception as e:
            print(f"Error sending data: {e}")           
        
    def server_close(self):
        self.client.close()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    # if client.connect():
    #     for i in range(10):  # 데이터 10번 전송
    #         data_to_send = f"Hello from client {i+1}"  # 예시 데이터
    #         if client.send_data(data_to_send):
    #             print(f"Sent message {i+1}")
    #         else:
    #             print(f"Failed to send message {i + 1}")
    #             break  # 전송 실패 시 루프 종료

    #     client.close()




    # from server_raspi import ServerRas

    # server = ServerRas() # 서버 객체 생성 (연결 수락)

    # for i in range(10):
    #     server.send_data(f"Message {i+1}")

    # server.close()