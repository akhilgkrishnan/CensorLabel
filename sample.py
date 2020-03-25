import cv2
img = cv2.imread("helmet.jpg")
y = 0
x = 0
h = 800
w = 600
crop_img = img[y:y+h, x:x+w]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)