from server_test import ServerTest
import time


food_list = ['rice', 'soup', 'aa', 'bb', 'cc']
weight_list = []
server_test = ServerTest()


# 실시간으로 값 가져오기

obj = None
cnt = 0
avg_weight = 0

while True:
    # obj 값이 들어오지 않았을 때는 그냥 실시간만 보여주고
    # 실시간 무게 값 수신
    onair_weight = server_test.get_weight()
    print(onair_weight)
    print(type(onair_weight))

    # obj 값이 들어오면 weight_list에 추가
    if obj is not None:
        weight_list.append(onair_weight)
        avg_weight = sum(weight_list) / len(weight_list)
        
        print("================================")
        print(f"weight_list = {weight_list}")
        print(f"avg_weight = {avg_weight}")
        print("================================")
        
    if cnt == 3:
        weight_list = []
        obj = 'rice'
    
    if cnt == 6:
        weight_list = []
        obj = 'soup'
    

    print(f"obj = {obj} \n onair_weight = {onair_weight} \n avg_weight = {round(avg_weight, 2)}")    
    time.sleep(2)
    cnt += 1
    


# for food in foodList:
    
#     result = server_test.get_weight_food(food)
#     print(f"result_rice = {result}")
#     time.sleep(5)
    
# while True:
#     result_soup = server_test.get_weight('soup')
#     print(f"result_rice = {result_soup}")
#     time.sleep(2)

# server_test.server_close()