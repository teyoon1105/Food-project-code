import pandas as pd
import numpy as np
from datetime import datetime


class FoodProcessor:
    def __init__(self, food_data_path, real_time_csv_path, customer_diet_csv_path, min_max_table):
        self.food_data = pd.read_csv(food_data_path, dtype={'food_id': str})
        self.real_time_csv_path = real_time_csv_path
        self.customer_diet_csv_path = customer_diet_csv_path
        self.min_max_table = min_max_table
        
    def get_food_info(self, food_id):
        food_info = self.food_data[self.food_data['food_id'] == food_id]
        if not food_info.empty:
            return food_info.to_dict(orient='records')[0]  # 첫 번째 레코드 반환
        else:
            print(f"[ERROR] Food ID {food_id} not found in data.")
            return None
        
    def calculate_nutrient(self, base_weight, base_value, consumed_weight):
        return base_value * (consumed_weight / base_weight)

    def load_min_max_table(file_path):
        try:
            min_max_table = pd.read_csv(file_path, encoding='utf-8', dtype={'food_id': str})
            return min_max_table
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV 파일 {file_path}을(를) 찾을 수 없습니다.")

    def calculate_q_ranges(self, food_id, min_max_table):
        food_info = min_max_table[min_max_table['food_id'] == str(food_id)]
        if food_info.empty:
            raise ValueError(f"음식 ID {food_id}에 대한 정보를 찾을 수 없습니다.")
        min_quantity = food_info['min'].values[0]
        max_quantity = food_info['max'].values[0]
        quantities = np.linspace(min_quantity, max_quantity, 5)
        return {f"Q{i+1}": quantities[i] for i in range(len(quantities))}

    def determine_q_category(self, measured_weight, q_ranges):
        q_values = list(q_ranges.values())
        for i in range(len(q_values) - 1):
            if q_values[i] <= measured_weight <= q_values[i + 1]:
                return f"Q{i+1}"
        return "Q5" if measured_weight > q_values[-1] else "Q1"

    def save_to_csv(self, file_path, data):
        try:
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, data], ignore_index=True)
        except FileNotFoundError:
            updated_df = data
        updated_df.to_csv(file_path, index=False)
        print(f"[INFO] Data saved to {file_path}.")

    def save_customer_diet_detail(self, customer_id, total_weight, total_nutrients):
        today_date = datetime.now().strftime('%Y%m%d')
        try:
            existing_data = pd.read_csv(self.customer_diet_csv_path)
            same_date_data = existing_data[
                (existing_data['customer_id'] == customer_id) &
                (existing_data['log_id'].str.startswith(f"{today_date}_{customer_id}"))
            ]
            log_number = len(same_date_data) + 1
        except FileNotFoundError:
            log_number = 1
            
        log_id = f"{today_date}_{customer_id}_{log_number}"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        new_data = {
            "log_id": [log_id],
            "customer_id": [customer_id],
            "total_weight": [total_weight],
            **{f"total_{key}": [round(value, 2)] for key, value in total_nutrients.items()},
            "timestamp": [timestamp]
        }
        self.save_to_csv(self.customer_diet_csv_path, pd.DataFrame(new_data))

    def process_food_data(self, plate_food_data, customer_id, min_max_table):
        if isinstance(min_max_table, str):
            min_max_table = pd.read_csv(min_max_table, dtype={'food_id': str})

        total_nutrients = {
            'calories': 0, 'carb': 0, 'fat': 0, 'protein': 0,
            'ca': 0, 'p': 0, 'k': 0, 'fe': 0, 'zn': 0, 'mg': 0
        }
        total_weight = 0
        real_time_food_info = []
        categories = {}

        for item in plate_food_data:
            try:
                food_id, measured_weight, measured_volume, region = item
            except ValueError:
                print(f"[ERROR] Failed to unpack item: {item}")
                continue

            food_info = self.food_data[self.food_data['food_id'] == food_id]
            if not food_info.empty:
                food_info_dict = food_info.to_dict(orient='records')[0]
                base_weight = food_info_dict['weight(g)']
                nutrients = {
                    'calories': food_info_dict['calories(kcal)'],
                    'carb': food_info_dict['carb(g)'],
                    'fat': food_info_dict['fat(g)'],
                    'protein': food_info_dict['protein(g)'],
                    'ca': food_info_dict['ca(mg)'],
                    'p': food_info_dict['p(mg)'],
                    'k': food_info_dict['k(mg)'],
                    'fe': food_info_dict['fe(mg)'],
                    'zn': food_info_dict['zn(mg)'],
                    'mg': food_info_dict['mg(mg)']
                }
                calculated_nutrients = {
                    key: self.calculate_nutrient(base_weight, value, measured_weight)
                    for key, value in nutrients.items()
                }

                try:
                    q_ranges = self.calculate_q_ranges(food_id, min_max_table)
                    category = self.determine_q_category(measured_weight, q_ranges)
                    categories[food_id] = category
                except ValueError:
                    print(f"[ERROR] Failed to determine category for food_id {food_id}")
                    categories[food_id] = "Unknown"

                real_time_food_info.append({
                    "food_id": str(food_id),
                    "name": food_info_dict['name'],
                    "volume": round(measured_volume, 2),
                    "weight": round(measured_weight, 2),
                    **{key: round(value, 2) for key, value in calculated_nutrients.items()},
                    "category": category,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

                for key in total_nutrients:
                    total_nutrients[key] += calculated_nutrients[key]
                total_weight += measured_weight
            else:
                print(f"[ERROR] Food ID {food_id} not found in food_data.")

        if real_time_food_info:
            self.save_to_csv(self.real_time_csv_path, pd.DataFrame(real_time_food_info))

        return total_nutrients
