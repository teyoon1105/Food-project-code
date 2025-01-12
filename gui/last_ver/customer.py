from datetime import datetime
import pandas as pd


# 고객정보 담당 & 기초대사랑 & 한끼 권장 식사량 게산클래스
class CustomerManager:
    def __init__(self, user_csv_file_path):
        self.user_csv_file_path = user_csv_file_path

    # 고객정보를 CSV파일에서 가져오는 함수
    def get_customer_info(self, customer_id):   # 고객아이디와 고객정보파일경로를 매개변수로 가져옴
        try:
            # csv파일 읽기(연락처를 문자열로 변환)
            customer_data = pd.read_csv(self.user_csv_file_path, dtype={'phone': object})
            # 매개변수로 주어진 고객아이디로 고객정보 필터링
            customer_info = customer_data[customer_data['id'] == customer_id]
            if not customer_info.empty:
                # orient : 출력한 dict의 형태를 지정 / records : [ { 열 : 값 , 열 : 값 }, { 열 : 값, 열 : 값 } ]
                return customer_info.to_dict(orient='records')[0]
            else:
                return None
        except FileNotFoundError:
            print(f"CSV 파일 {self.user_csv_file_path}이(가) 존재하지 않습니다.")
            return None

    # 만나이 계산
    def calculate_age(self, birth_date):    # 8자리의 생년월일 
        
        today = datetime.now()
        birth_year, birth_month, birth_day = int(birth_date[:4]), int(birth_date[4:6]), int(birth_date[6:])
        # 현재 날짜와 비교하여 만 나이 계산
        age = today.year - birth_year - ((today.month, today.day) < (birth_month, birth_day))
        return age
    
    # bmr(기초대사량) 게산 
    def calculate_bmr(self, weight, height, age, gender, exercise_score):
        # Mifflin-st-jeor 공식으로 BMR 계산
        bmr = (10 * weight) + (6.25 * height) - (5 * age)
        # 성별에 따라 BMR 조정
        if gender == 'F':
            bmr -= 161
        elif gender == 'M':
            bmr += 5
        else:
            raise ValueError("성별은 'M' 또는 'F'로 입력해야 합니다.")

        activity_multiplier = {
            'A': 1.9, 'B': 1.725, 'C': 1.55, 'D': 1.375, 'E': 1.2
        }
        # 활동량에 따라 BMR에 활동량 계수를 곱해줌
        # 운동량 정보가 없으면(이제 막 가입한 회원은 운동량 정보가 없을 수 있음) 기본값 1.55를 곱해주자
        return bmr * activity_multiplier.get(exercise_score, 1.55)
    
    
    def calculate_bmi(self, weight, height):
        if height > 10:  # 키가 cm 단위로 제공되는 경우
            height = height / 100  # cm -> m 변환
            return weight / (height ** 2)
    
    
    def recommend_nutrient_ratio(self, bmi):
        """
        BMI에 따라 탄단지 비율을 추천하는 함수
        :param bmi: BMI 값
        :return: 추천 비율 (탄수화물, 단백질, 지방)
        """
        if 18.5 <= bmi <= 24.9:
            return [50, 30, 20]  # 일반인 비율
        elif bmi >= 25:
            return [40, 40, 20]  # 다이어트 필요 비율
        else:
            return None  # 비정상 값 (저체중 등)
        
        
    