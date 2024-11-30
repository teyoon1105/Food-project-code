## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##          rs400 advanced mode tutorial           ##
# 리얼센스 카메라의 고급 모드를 설정 및 제어하는 코드
#####################################################

# 카메라 제어 하기 위한 라이브러리
# First import the library
import pyrealsense2 as rs
# 시간 지연을 위한 타임 모듈
import time
# json 처리를 위한 json모듈
import json

# D400 시리즈 카메의 제품 ID 리스트 정의
# 고급 모드 지원하는 디바이스 식별하는데 사용
DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C", "0B5B"]

# 고급 모드 지원 디바이스 찾는 함수
def find_device_that_supports_advanced_mode() :
    # 카메라 검색을 위한 시스템 컨텍스트
    ctx = rs.context()
    # 기본 디바이스 객체 생성
    ds5_dev = rs.device()
    # 연결된 모든 디바이스 조회
    devices = ctx.query_devices()

    # 조회된 디바이스 확인
    for dev in devices:
        # 디베이스가 제품 ID 정보를 지원하고, 해당 ID가 리스트 목록에 있는 지 확인
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
            return device
    # 적합한 디바이스를 찾지 못한 경우 예외 발생
    raise Exception("No D400 product line device that supports advanced mode was found")

try:
    # 고급 모드 활성화
    # 사용 가능한 디바이스 가져옴
    dev = find_device_that_supports_advanced_mode()
    # 고급 모드 객체 생성
    advnc_mode = rs.rs400_advanced_mode(dev)
    print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

    # Loop until we successfully enable advanced mode
    # 고급 모드가 활성화 될 때까지 반복
    while not advnc_mode.is_enabled():
        print("Trying to enable advanced mode...")
        # 고급 모드 활성화 시도
        advnc_mode.toggle_advanced_mode(True)
        # At this point the device will disconnect and re-connect.
        print("Sleeping for 5 seconds...")
        # 활성화 후 디바이스 재연결을 위하 5초 대기
        time.sleep(5)

        # 재연결 한 후 디바이스 객체 초기화
        # The 'dev' object will become invalid and we need to initialize it again
        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

    # 카메라 설정값들 조회
    # Get each control's current value
    # 깊이 제어
    # 깊이 측정의 정확도와 노이즈 제거에 영향
    print("Depth Control: \n", advnc_mode.get_depth_control())
    
    # Rolling Shutter Mode
    # 롤링 셔터 모드 관련 설정
    # 이미지 캡쳐 타이밍과 동기화 제어
    # 움지이는 물체 촬용시 왜곡 감소에 사용
    print("RSM: \n", advnc_mode.get_rsm())
    
    # Range Auto-exposure Unit
    # 자동 노출 조정을 위한 범위 설정
    # 거리에 따른 노출 보정
    # 다양한 거리의 물체를 잘 인식하도록 조정
    print("RAU Support Vector Control: \n", advnc_mode.get_rau_support_vector_control())
    
    # 컬러 제어
    # 색상 이미지의 채도, 대비, 밝기 조정
    print("Color Control: \n", advnc_mode.get_color_control())
    
    # 노출 제어
    # 자동 노출 조정을 위한 범위 설정
    # 거리에 따른 노출 보정
    # 다양한 거리의 물체를 잘 인식하도록 조정
    print("RAU Thresholds Control: \n", advnc_mode.get_rau_thresholds_control())
    
    # Spatial Left-right Overdrive
    # 좌우 이미지 매칭 품질 향상
    # 스테레오 비너의 정확도 개선
    # 잘못된 매칭에 대한 패널티 설정
    print("SLO Color Thresholds Control: \n", advnc_mode.get_slo_color_thresholds_control())
    print("SLO Penalty Control: \n", advnc_mode.get_slo_penalty_control())
    
    # High Detail All Direction
    # 모든 방향의 고세부 정보 처리
    # 깊이 맵의 세부 디테일 향상
    # 엣지 보존 및 노이즈 제거
    print("HDAD: \n", advnc_mode.get_hdad())
    
    # 색상 보정
    # 컬러 이미지의 색상 보정
    # 화이트 밸러스, 감마 조정
    # 실제 색상에 가깝게 고정
    print("Color Correction: \n", advnc_mode.get_color_correction())
    
    # 
    print("Depth Table: \n", advnc_mode.get_depth_table())
    
    # 자동 노출 동작 방식 설정
    # 밝기 조건 변화에 대한 반응 조정
    # 노출 시간 제어
    print("Auto Exposure Control: \n", advnc_mode.get_ae_control())

    # 센서스 변환
    # 스테레오 매칭을 위한 센서스 변환 파라미터
    # 깊이 계산의 기본이 되는 픽셀 비교 방식 설정
    print("Census: \n", advnc_mode.get_census())

    # 각 컨트롤의 최소/최대 기능 값 조회
    #To get the minimum and maximum value of each control use the mode value:
    query_min_values_mode = 1
    query_max_values_mode = 2

    # 현재 설정된 값 조회
    # 최소, 최대 깊이 제어 설정 가져오기
    current_std_depth_control_group = advnc_mode.get_depth_control()
    min_std_depth_control_group = advnc_mode.get_depth_control(query_min_values_mode)
    max_std_depth_control_group = advnc_mode.get_depth_control(query_max_values_mode)
    print("Depth Control Min Values: \n ", min_std_depth_control_group)
    print("Depth Control Max Values: \n ", max_std_depth_control_group)

    # 새로운 값 설정
    # 특정 컨트롤 값을 최소/최대값의 중간값으로 설정
    # Set some control with a new (median) value
    current_std_depth_control_group.scoreThreshA = int((max_std_depth_control_group.scoreThreshA - min_std_depth_control_group.scoreThreshA) / 2)
    advnc_mode.set_depth_control(current_std_depth_control_group)
    print("After Setting new value, Depth Control: \n", advnc_mode.get_depth_control())

    # JSON 직렬화/역직렬화
    # 모든 컨트롤 설정을 json 문자열로 직렬화
    # Serialize all controls to a Json string
    serialized_string = advnc_mode.serialize_json()
    print("Controls as JSON: \n", serialized_string)
    # json 문자열을 파이썬 객체로 변환
    as_json_object = json.loads(serialized_string)

    # 모든 컨트롤 설정을 JSON으로 저장
    # JSON을 통해 설정 로드 가능
    # We can also load controls from a json string
    # For Python 2, the values in 'as_json_object' dict need to be converted from unicode object to utf-8
    # 파이썬 2에 알맞게 문자열 인코딩
    if type(next(iter(as_json_object))) != str:
        as_json_object = {k.encode('utf-8'): v.encode("utf-8") for k, v in as_json_object.items()}
    # The C++ JSON parser requires double-quotes for the json object so we need
    # to replace the single quote of the pythonic json to double-quotes
    # 파이썬의 작은 따옴표를 json 표준인 큰 따옴표로 변환
    json_string = str(as_json_object).replace("'", '\"')
    # json 설정을 카메라에 로드
    advnc_mode.load_json(json_string)

except Exception as e:
    print(e)
    pass
