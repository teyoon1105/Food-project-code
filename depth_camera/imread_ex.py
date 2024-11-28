import cv2

img_path = "C:/Users/SBA/Pictures/Screenshots/foodtray.png"

img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not read image from {img_path}")
else:
    cv2.namedWindow('img')
    cv2.imshow('img', img)

    key = cv2.waitKey(0)

    if key == 27:  # Check for ASCII value of Escape key (27)
        cv2.destroyAllWindows()
    elif key == ord('q'): #or if you want to close by pressing 'q'
        cv2.destroyAllWindows()