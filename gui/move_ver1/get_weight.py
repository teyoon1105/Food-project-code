from server_client import Client
from threading import Thread
import threading

# 스레드간 데이터 공유를 위한 변수 및 락
recent_weight_str = "0"
weight_list = [0,0]
object_weight_list = [0, 1]
stop = False
first_object_weight = 0.0
object_weight = 0.0
obj_name = ''

weight_lock = threading.Lock()

# onair Thread
def onair(client):
    global recent_weight_str
    global object_weight_list
    global weight_list
    global obj_name
    global first_object_weight
    global object_weight
    # client = Client('172.31.99.10', 5001) # 라즈베리파이 IP 주소 사용
    # client = Client('192.168.191.100', 5000) # 라즈베리파이 IP 주소 사용
    
    # client.connect()
    
    # recent_weight = 0
    try:
        while client.connected:
            print("오긴오니")

            onair_weight_str = client.receive_data()
            if onair_weight_str == 'False':
                onair_weight_str = weight_list[-2]

            elif onair_weight_str is None:
                # recent_weight = -100
                print("Failed connect Thread")
                # break

            with weight_lock:
                recent_weight_str = onair_weight_str
                
                # 더해진 가장 최근 무게값(소수점 둘째자리)
                recent_weight = round(float(recent_weight_str), 2)

                # detect된 객체의 무게값
                # object_weight = recent_weight - first_object_weight

                weight_list.append(recent_weight)

            print(f"onair : {recent_weight}")
            # print(f"{obj_name} : {object_weight}")
            print("=================================")
    except Exception as e:
        return print(e)

def get_object_weight():
    global object_weight
    
    print(f'하이:{object_weight}')
    return object_weight


class GetWeight():
    def __init__(self):
        # self.client = Client('172.30.1.100', 5000) # 라즈베리파이 IP 주소 사용
        # self.client = Client('172.31.99.10', 5000) # 라즈베리파이 IP 주소 사용
        self.client = Client('192.168.1.4',5000)
        # self.client = Client('192.168.191.100', 5000) # 라즈베리파이 IP 주소 사용
        # self.client = Client('192.168.45.150', 5000) 
        # self.client.connect()
        
        
        if self.client.connect():
            self.weight_thread = Thread(target=onair, args=(self.client,))
            self.weight_thread.daemon = True
            self.weight_thread.start()
        else:
            print("Failed to connect to the server.")
        
        
        
    def get_weight(self):
        global recent_weight_str
        with weight_lock:
            weight_str = recent_weight_str
        try:
            # if self.client.connect():
            weight = float(weight_str)
            # if weight is None:
                # print("Failed to receive data. Closing connection.")
            return f"{round(weight, 2)}"
        except Exception as e:
            return print(f"Error : {e}")
        
    def send_object(self, name):
        if self.client.connected:
            self.client.send_data(name)
            print("send Data")
            return True
        





