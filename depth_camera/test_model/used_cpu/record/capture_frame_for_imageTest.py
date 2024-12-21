import cv2
import os
import time

# ---- 전역 변수 ----
video_path = "C:/Users/SBA/teyoon_github/Food-project-code/depth_camera/test_model/record/test_model.avi"  # 저장된 영상 파일 경로
save_directory = "saved_frames"  # 프레임 저장 폴더
os.makedirs(save_directory, exist_ok=True)  # 저장 폴더 생성

current_frame = None  # 현재 프레임 저장 변수
paused = False  # 영상 일시정지 상태


def mouse_callback(event, x, y, flags, param):
    """마우스 클릭 이벤트 처리"""
    global current_frame
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭 시
        if current_frame is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")  # 시간 기반 이름
            save_path = os.path.join(save_directory, f"frame_{timestamp}.jpg")
            cv2.imwrite(save_path, current_frame)
            print(f"Frame saved at: {save_path}")


def main():
    global current_frame, paused

    # 비디오 파일 불러오기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open the video file.")
        return

    cv2.namedWindow('Video Player')  # 창 생성
    cv2.setMouseCallback('Video Player', mouse_callback)  # 마우스 이벤트 연결

    while cap.isOpened():
        if not paused:  # 영상 재생 상태
            ret, frame = cap.read()
            if not ret:  # 영상 끝
                print("End of video.")
                break
            current_frame = frame.copy()  # 현재 프레임 저장
            cv2.imshow('Video Player', frame)  # 영상 출력

        key = cv2.waitKey(30)  # 30 FPS 기준으로 딜레이 조정

        if key == 27:  # ESC 키로 종료
            break
        elif key == ord('p'):  # P 키로 일시정지/재생
            paused = not paused
            if paused:
                print("Video paused. Press 'P' to resume.")
            else:
                print("Video resumed.")
        elif key == ord('q'):  # Q 키로 종료
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")


if __name__ == "__main__":
    main()
