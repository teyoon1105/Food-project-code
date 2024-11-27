## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#####################################################
## librealsense tutorial #1 - Accessing depth data ##
## ASCII 아트 형태로 깊이 맵을 터미널에 출력하는 코드
#####################################################

# First import the library
# 인텔 리얼센스 카메라를 제어하기 위한 python 라이브러리
import pyrealsense2 as rs

try:
    # Create a context object. This object owns the handles to all connected realsense devices
    # 파이프라인 
    # : 카메라의 데이터 스트림을 관리하는 객체
    # : 카메라로부터 프레임을 받아오는 통로
    pipeline = rs.pipeline()

    # Configure streams
    # 카메라 설정 구성
    # depth 스트림 활성화
    # 해상도 설정(640, 480)
    # 포맷(16비트 깊이 데이터)
    # FPS 30프레임
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    # 위의 구성으로 카메라 스트리밍 시작
    pipeline.start(config)


    # 무한 루프를 돌면서
    while True:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        
        # 새로운 프레임이 준비될 때 까지 대기
        frames = pipeline.wait_for_frames()
        # 깊이 프레임을 가쟈옴
        depth = frames.get_depth_frame()
        # 깊이 프레임이 없으면 다음 반복으로 넘어감
        if not depth: continue

        # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and approximating the coverage of pixels within one meter
        # 깊이 데이터를 시작화하는 부분
        # 화면을 64개 구역으로 나눔(가로기준 64개 구역으로)
        coverage = [0]*64
        for y in range(480):
            for x in range(640):
                # 각 픽셀들의 거리를 측정하고
                dist = depth.get_distance(x, y)
                # 픽셀들 중 1m 거리에 있는 픽셀들을 카운트
                if 0 < dist and dist < 1:
                    coverage[x//10] += 1
            # 20 줄마다 한번씩 시각화 결과 출력
            if y%20 is 19:
                line = ""
                for c in coverage:
                    # 공백(가장 멀음)부터 W(가장 가까움)까지 8단계로 표현
                    line += " .:nhBXWW"[c//25]
                # 초기화하고 다음 줄로 넘어감
                coverage = [0]*64
                print(line)
    exit(0)
#except rs.error as e:
#    # Method calls agaisnt librealsense objects may throw exceptions of type pylibrs.error
#    print("pylibrs.error was thrown when calling %s(%s):\n", % (e.get_failed_function(), e.get_failed_args()))
#    print("    %s\n", e.what())
#    exit(1)

# 오류 처리
except Exception as e:
    print(e)
    pass
