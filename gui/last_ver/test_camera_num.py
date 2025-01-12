import cv2

def check_camera():
    for cam_index in range(5):  # 최대 5개의 카메라를 확인 (필요시 숫자 조정)
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            print(f"Camera {cam_index} is available.")
            cap.release()
        else:
            print(f"Camera {cam_index} is not available.")

def display_camera_feed(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}.")
        return

    print(f"Displaying feed from camera {camera_index}. Press 'ESC' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow(f"Camera {camera_index} Feed", frame)
        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_camera()  # 사용 가능한 카메라 번호 확인
    camera_index = int(input("Enter the camera index to test: "))
    display_camera_feed(camera_index)
