import cv2 as cv2
import numpy as np
def add_warning(image,logo,alpha=1,scale=0.5,y=-120,x=20):
    logo = cv2.imread(logo, cv2.IMREAD_UNCHANGED)
    h, w = image.shape[:2]
    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
    overlay = cv2.resize(logo, None,fx=scale,fy=scale)
    (wH, wW) = overlay.shape[:2]
    output = image.copy()
    # blend the two images together using transparent overlays
    try:
        if x<0 : x = w+x
        if y<0 : y = h+y
        if x+wW > w: wW = w-x  
        if y+wH > h: wH = h-y
        overlay=cv2.addWeighted(output[y:y+wH, x:x+wW],alpha,overlay[:wH,:wW],1.0,0,overlay[:wH,:wW])
        output[y:y+wH, x:x+wW ] = overlay
    except Exception as e:
        print("Error: Logo position is overshooting image!")
        print(e)
    output= output[:,:,:3]
 
    return output

