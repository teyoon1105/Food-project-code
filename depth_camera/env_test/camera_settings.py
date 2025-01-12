import pyrealsense2 as rs

# Camera parameter check code
# 카메라 초점거리(하드웨어값) 변수값으로 사용하기 위해 확인

def extract_camera_intrinsics():
    pipeline = rs.pipeline()
    # 카메라 설정 객체 
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    try:
        pipeline.start(config)
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # 문제없이 깊이 프레임이 들어온다면 
        if depth_frame:
            # 내부 파라미터 확인
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            print("Camera Intrinsics:")
            print(f"  fx: {intrinsics.fx}") # x축 초점거리
            print(f"  fy: {intrinsics.fy}") # y축 초점거리
            print(f"  ppx: {intrinsics.ppx}") # 광학중심의 x좌표
            print(f"  ppy: {intrinsics.ppy}") # 광학중심의 y좌표
            print(f"  width: {intrinsics.width}") # 프레임의 너비
            print(f"  height: {intrinsics.height}") # 프레임의 높이

        else:
            print("Failed to retrieve depth frame.")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    extract_camera_intrinsics()