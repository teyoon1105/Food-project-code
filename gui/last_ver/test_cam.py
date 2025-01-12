import cv2

def webcam_test():
    # 웹캠 초기화 (기본 카메라: 0번)
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press 'q' to exit.")

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame from the webcam.")
            break

        # 화면에 프레임 표시
        cv2.imshow("Webcam Test", frame)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_test()