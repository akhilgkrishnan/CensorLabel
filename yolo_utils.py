import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from PIL import Image 


def cv2_to_pil(img): #Since you want to be able to use Pillow (PIL)
    return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))    
def add_label(img,height,dtype):
    logo = Image.open(dtype)
    pil_img = cv2_to_pil(img)
    logo = logo.convert("RGBA")
    logo = logo.resize((250,40))
    image_copy = pil_img.copy()
    position = (10,height-65)
    image_copy.paste(logo, position,logo)
    image_copy.save("pasted_image.jpg")
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    intersectionArea = (xB - xA) * (yB - yA)
    unionArea = (boxA[2]*boxA[3])+(boxB[2]*boxB[3])-intersectionArea;
    overlapArea = intersectionArea/unionArea;
    return overlapArea    
 
    
    
def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels,height,frameCount):
    # If there are any detections
    detect = 0
    if len(idxs) > 0:
        
        print("idxs :",idxs.flatten())
        motorcycle = list(filter(lambda x: classids[x] == 0,idxs.flatten()))
        helmet = list(filter(lambda x: classids[x] == 2,idxs.flatten()))
        whelmet = list(filter(lambda x: classids[x] == 1,idxs.flatten()))

        print("Motorcycle :",motorcycle)
        print("helmet :",helmet)
        print("whelmet :",whelmet)

        if len(motorcycle) != 0:
            for i in motorcycle:
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                motorBox = [x,y,w,h]
                if len(helmet) != 0:
                    for j in helmet:
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        helmetBox = [x,y,w,h]
                        iou = bb_intersection_over_union(motorBox,helmetBox)
                        print("iou:",iou)
                        if iou > 0.0:
                            print("Person wear helmet")
                        else:
                            print("person not weared helmet")    
                if len(whelmet) != 0:
                    for j in whelmet:
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        whelmetBox = [x,y,w,h]
                        iou = bb_intersection_over_union(motorBox,whelmetBox)
                        if iou > 0:
                            print("Person not wear helmet")
        for i in idxs.flatten():     
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]   
            color = [int(c) for c in colors[classids[i]]]
            
            #Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            #Adding "smoking injurious to health" label to each smoking detected frame

            # if 0 in classids: #Check the detected item is smoking
            #     add_label(img,height,'smoke.png')
            #     labelledImg = cv.imread("pasted_image.jpg")
            #     detect = 1
            # elif 1 in classids:
            #     labelledImg = cv.imread("pasted_image.jpg")
            #     detect =2
            # else:
            #     labelledImg = img
            # return labelledImg,detect    
    cv.imshow("frame",img)
    key = cv.waitKey(1) & 0xFF
    return img,5

def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')
            
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            # Consider only the predictions that are above a certain confidence level
            if confidence > 0.3:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS,frameCount, 
            boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    
    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                       

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        # if FLAGS.show_time:
        #     print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

        
        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, FLAGS.confidence)
        
        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
        
    # Draw labels and boxes on the image
    img,detect = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels,height,frameCount)
    
    return img, detect
