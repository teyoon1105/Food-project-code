import cv2, sys
import numpy as np
import os
from glob import glob

# data 폴더에서 파일 목록 읽기 > 리스트로 뽑아내기
# 이미지 불러오기
# 마우스 콜백 함수 생성
# 콜백함수 안에서 박스를 그리고, 박스 좌표를 뽑아낸다. (마우스 좌표 2개)
# 참고 : YOLO에서는 박스의 중심좌표(x,y), w, h 사용
# 이미지 파일명과 동일한 파일명으로 txt 파일을 생성
# 추가 : 화살표(>)를 누르면 다음 이미지 로딩됨
# 추가 : 화살표(<)를 누르면 txt 파일이 있다면 박스를 이미지 위에 띄워주기
# 추가 : 박스를 잘못쳤을 때 c(clear)를 누르면 현재 박스 초기화


# 파일을 불러오기 위한 함수
def getImageList():
    
    basePath = os.getcwd()
    
    dataPath = os.path.join(basePath, 'images')
   
    fileNames = glob(os.path.join(dataPath, '*.jpeg'))

   
    return fileNames


def drawROI(image, corners):  
   
    cpy = img.copy()
    color = (128,128,256)
    thick = 3
    # print(corners)
    
    for corners in boxList:
        cv2.rectangle(cpy, tuple(corners[0]), tuple(corners[1]), color = color, thickness = thick)
    
    
    disp = cv2.addWeighted(img, 0.3, cpy, 0.7,0)
    
    
    return disp



def on_mouse(event, x, y, flags, param):
    global startPt, img, ptList, cpy, boxList, textWrData
    
    
    if event == cv2.EVENT_LBUTTONDOWN:
        startPt = (x,y)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if startPt:
            ptList=[startPt, (x,y)]
            boxList.append(ptList)
            cpy = drawROI(img, boxList)
            boxList.pop()
            cv2.imshow('label', cpy) 
    
    elif event == cv2.EVENT_LBUTTONUP:
       
        ptList = [startPt, (x,y)]
        boxList.append(ptList)
        textWrData = str(ptList)
         
        cpy = drawROI(img, boxList) 
        
        startPt = None
        
        cv2.imshow('label', cpy) 
    
def load_box(filename):
    txtfileName = filename + '.txt'
    box_data = []
    if os.path.exists(txtfileName):
        with open(txtfileName, 'r') as f:
            data = eval(f.read())
            box_data = [list(map(tuple, box)) for box in data]
    return box_data

def saveBoxes(filename, boxes):
    txtfileName = filename + '.txt'
    f = open(txtfileName, 'w')
    f.write(str(boxes))
    print('file saved : {}'.format(txtfileName))  
        
        
# main함수
ptList = [] 
startPt = None
cpy=[]
boxList = []
textWrData = ''
current_index = 0
    

fileNames = getImageList()

img = cv2.imread(fileNames[current_index])
boxList = load_box(os.path.splitext(fileNames[current_index])[0])
cv2.namedWindow('label')
cv2.setMouseCallback('label', on_mouse, [img])
cv2.imshow('label', img)


while True:
   
    key = cv2.waitKeyEx()
    if key == 27:
        break
    elif key == 13:
        saveBoxes(os.path.splitext(fileNames[current_index])[0], boxList)
        
    elif key == ord('c'):
        boxList = []
        cv2.imshow('label', img)
        
    elif key == 83:
        if current_index < len(fileNames)-1:
            current_index += 1
            
        else:
            current_index = 0
            
        img = cv2.imread(fileNames[current_index])
        boxList = load_box(os.path.splitext(fileNames[current_index])[0])
        cv2.imshow('label', drawROI(img, boxList))
        cv2.setMouseCallback('label', on_mouse, [img])
        
    elif key == 81:
        if current_index > 0:
            current_index -= 1
        else:
            current_index = len(fileNames)-1
        
        img = cv2.imread(fileNames[current_index])
        boxList = load_box(os.path.splitext(fileNames[current_index])[0])
        cv2.imshow('label', drawROI(img, boxList))
        cv2.setMouseCallback('label', on_mouse, [img])

        
   
boxList = []
cv2.destroyAllWindows()