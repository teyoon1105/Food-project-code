import cv2

# opencv library test 코드
# img 위에 개행 효과를 사용하여 putText

img_path = "path/your/img"
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

