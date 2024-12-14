import cv2

img_path = "D:/food_detection/train_16/brown seaweed_soup/04011005_train/04_041_04011005_160273319870986.jpeg"

img = cv2.imread(img_path)


text = "This is \nsome text"
y0, dy = 50, 50
for i, line in enumerate(text.split('\n')):
    y = y0 + i*dy
    cv2.putText(img, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

cv2.imshow('Depth Image', img)

key = cv2.waitKey(0)

if key == 27:
    cv2.destroyAllWindows

