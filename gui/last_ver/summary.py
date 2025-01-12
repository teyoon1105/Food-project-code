import pandas as pd
from customer import CustomerManager


# 3개의 섹션으로 구성되어 있음 

# 1. 칼로리와 권장칼로리를 비교하는 BAR CHART 
# 칼로리 : customer_diet_detail.csv에 저장된 가장 마지막 행에서 칼로리 열의 값을 가져온다.
# 권장칼로리 : CustomerManager클래스의 caculator_bmr의 반환값을 가져와서 1/3로 나눠서 보여준다.


# 2. 섭취된 탄단지 비율과 권장 탄단지 비율을 비교하는 PIE CHART
# 영양소 값
# carb = 4.97   # 탄수화물 (g)
# fat = 9.74    # 지방 (g)
# protein = 8.26 # 단백질 (g)


# 권장 탄단지 
# 고객의 bmi를 계산해서(키와 몸무게등으로 계산할 수 있음)
# bmi = 체중/키
# BMI에 따른 비율 설정
# 일반인: BMI가 18.5 이상 24.9 이하 → 비율 5:3:2 (탄수화물 50%, 단백질 30%, 지방 20%).
# 다이어트 필요: BMI가 25 이상 → 비율 4:2:2 (탄수화물 40%, 단백질 20%, 지방 20%).

# BMI 계산
# bmi = calculate_bmi(weight, height)
# print(f"BMI: {bmi:.2f}")

# 비율 추천
# recommended_ratio = recommend_nutrient_ratio(bmi)


# 3. 주요 영양성분(칼슘,칼륨,철분,아연,마그네슘) 섭취량을 보여주는 원형진행바 
# 성인 하루 권장량의 1/3을 100으로 두고 그 중 해당 식사에서 몇퍼센트를 섭취하였는지 미리 계산해두고 퍼센테이지를 UI로 가져간다. 


import pandas as pd
from customer import CustomerManager


class NutritionSummary:
    def __init__(self, user_csv_file_path, diet_csv_file_path, customer_id):
        # NutritionSummary 클래스 초기화
        self.customer_manager = CustomerManager(user_csv_file_path)  # CustomerManager 객체 생성
        self.diet_info = pd.read_csv(diet_csv_file_path)  # 식단 CSV 파일 읽기
        self.customer_id = customer_id
        # self.customer_id = self.diet_info['customer_id'].iloc[-1]  # 현재 고객 ID

        # 고객 정보 가져오기
        self.customer_info = self._load_customer_info(user_csv_file_path)
        self.weight = self.customer_info['weight']
        self.height = self.customer_info['height']
        self.birth = self.customer_info['birth']
        self.age = self.customer_manager.calculate_age(self.birth)
        self.gender = self.customer_info['gender']
        self.exercise = self.customer_info['exercise']

    # 고객정보 로드
    
    # def _load_customer_info(self, user_csv_file_path):
    #     customer_path = pd.read_csv(user_csv_file_path, dtype= {'birth': str})
    #     customer_id = self.customer_id
    #     customer_info = customer_path[customer_path['id'] == customer_id].iloc[0].to_dict()
    #     return {
    #         'name': customer_info['name'],
    #         'weight': customer_info['weight'],
    #         'height': customer_info['height'],
    #         'birth': customer_info['birth'],
    #         'gender': customer_info['gender'],
    #         'exercise': customer_info['exercise']
    #     }


    def _load_customer_info(self, user_csv_file_path):
        # CSV 파일 로드
        customer_path = pd.read_csv(user_csv_file_path, dtype={'birth': str})
        
        # customer_id를 사용하여 해당 고객 정보를 가져오기
        filtered_data = customer_path[customer_path['id'] == self.customer_id]

        if not filtered_data.empty:
            customer_info = filtered_data.iloc[0].to_dict()
            return {
                'name': customer_info['name'],
                'weight': customer_info['weight'],
                'height': customer_info['height'],
                'birth': customer_info['birth'],
                'gender': customer_info['gender'],
                'exercise': customer_info['exercise']
            }
        else:
            print(f"Customer ID {self.customer_id} not found.")
            return None
        

    # 대시보드 최상단에 고객의 이름을 기입하기위해서 이름만 추출
    def get_customer_name(self):
        print(f"고객명: {self.customer_info['name']}")
        return self.customer_info['name']  # 'name' 열에서 이름 추출
    
    # 섭취칼로리,권장칼로리,한끼권장칼로리 값 가져오기
    def get_calories_data(self):
        consumed_calories = self.diet_info['total_calories'].iloc[-1]
        recommend_calories = self.customer_manager.calculate_bmr(
            self.weight, self.height, self.age, self.gender, self.exercise)
        one_meal_recommend_calories = round(recommend_calories / 3, 2)
        return consumed_calories, recommend_calories, one_meal_recommend_calories
    
    # bmi 가져오기 : 권장 탄단지 비율을 확인하기 위함
    def get_bmi(self):
        return self.customer_manager.calculate_bmi(self.weight, self.height)

    # bmi에 따른 탄단지 비율 계산하기(2가지임)
    def get_recommended_nutrient_ratio(self):
        bmi = self.get_bmi()
        return self.customer_manager.recommend_nutrient_ratio(bmi)

    # 섭취한 탄단지를 비율로 변환
    def get_consumed_nutrient_ratio(self):
        carb = self.diet_info['total_carb'].iloc[-1]
        protein = self.diet_info['total_protein'].iloc[-1]
        fat = self.diet_info['total_fat'].iloc[-1]

        total = carb + protein + fat
        return {
            'carbohydrates': round((carb / total) * 100, 1),
            'proteins': round((protein / total) * 100, 1),
            'fats': round((fat / total) * 100, 1)
        }

    # 칼슘,칼륨,철분,아연,마그네슘 권장섭취량 대비 섭취율 계산
    def get_nutrient_percentages(self):
        
        # 한끼 권장섭취량
        main_nutrient_recommend = {
            'ca': 400,  # 칼슘
            'k': 1200,  # 칼륨
            'fe': 330,  # 철분
            'zn': 3,  # 아연
            'mg': 110  # 마그네슘
        }
        
        # 섭취량 가져오기
        consumed_nutrient = {
            'ca': self.diet_info['total_ca'].iloc[-1],
            'k': self.diet_info['total_k'].iloc[-1],
            'fe': self.diet_info['total_fe'].iloc[-1],
            'zn': self.diet_info['total_zn'].iloc[-1],
            'mg': self.diet_info['total_mg'].iloc[-1],
        }

        percentages = {}
        for nutrient, recommended_value in main_nutrient_recommend.items():
            consumed_value = consumed_nutrient.get(nutrient, 0)
            percentages[nutrient] = round((consumed_value / recommended_value) * 100, 1)
        return percentages



# 디버깅용 실행 코드
if __name__ == "__main__":
    user_csv_file_path = "FOOD_DB/user_info.csv"
    diet_csv_file_path = "FOOD_DB/customer_diet_detail.csv"

    # NutritionSummary 객체 생성
    nutrition_summary = NutritionSummary(user_csv_file_path, diet_csv_file_path)

    # 칼로리 데이터 가져오기
    consumed_calories, recommend_calories, one_meal_recommend_calories = nutrition_summary.get_calories_data()
    print(f"섭취 칼로리: {consumed_calories}")
    print(f"권장 칼로리: {recommend_calories}")
    print(f"한 끼 권장 칼로리: {one_meal_recommend_calories}")

    # BMI와 권장 탄단지 비율 가져오기
    bmi = nutrition_summary.get_bmi()
    recommended_ratio = nutrition_summary.get_recommended_nutrient_ratio()
    print(f"BMI: {bmi:.2f}")
    print(f"권장 탄단지 비율: {recommended_ratio}")

    # 섭취 탄단지 비율 가져오기
    consumed_ratios = nutrition_summary.get_consumed_nutrient_ratio()
    print(f"섭취 탄단지 비율: {consumed_ratios}")

    # 주요 영양소 섭취 비율 가져오기
    nutrient_percentages = nutrition_summary.get_nutrient_percentages()
    print(f"주요 영양소 섭취 비율: {nutrient_percentages}")
