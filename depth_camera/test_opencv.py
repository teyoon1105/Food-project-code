import cv2

img_path = "D:/food_detection/train_16/brown seaweed_soup/04011005/04_041_04011005_160273319870986.jpeg"

img = cv2.imread(img_path)

cv2.imshow('Depth Image', img)

key = cv2.waitKey(0)

if key == 27:
    cv2.destroyAllWindows

