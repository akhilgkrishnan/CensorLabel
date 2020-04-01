import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os

riding = ['motorcycling','riding a bike','riding scooter','riding mountain bike']
smoking = ['smoking','smoking hookah']
#Determine the IOU of two bounding boxes
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
 
    
    
def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels,height,labelh):
    # If there are any detections
    detect = 0
    if len(idxs) > 0:
        
        print("idxs :",idxs.flatten())

        motorcycle = list(filter(lambda x: classids[x] == 0,idxs.flatten()))
        if (labelh in riding) and len(motorcycle) > 0:
            whelmet = list(filter(lambda x: classids[x] == 1,idxs.flatten()))
            helmet = list(filter(lambda x: classids[x] == 2,idxs.flatten()))
            
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
                            if iou < 0.25:
                                detect = 0 #person weared helmet
                              

                    if len(whelmet) != 0:
                        for j in whelmet:
                            x, y = boxes[i][0], boxes[i][1]
                            w, h = boxes[i][2], boxes[i][3]
                            whelmetBox = [x,y,w,h]
                            iou = bb_intersection_over_union(motorBox,whelmetBox)
                            print("iou :",iou)
                            if iou < 0.25:
                                detect = 1 #person not weared helmet
                                break
                return detect               
        elif labelh in smoking:
            smoke = list(filter(lambda x: classids[x] == 3,idxs.flatten()))
            if len(smoke) > 0:
                detect = 2
            else:
                detect = 0
        else:
            return detect          
                                     
                                
        # for i in idxs.flatten():     
        #     x, y = boxes[i][0], boxes[i][1]
        #     w, h = boxes[i][2], boxes[i][3]   
        #     color = [int(c) for c in colors[classids[i]]]
            
        #     #Draw the bounding box rectangle and label on the image
        #     cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
        #     text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
        #     cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

             
        # cv.imshow("frame",img)
        # key = cv.waitKey(1) & 0xFF
    else:
        return detect

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

def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS,labelh, 
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
        print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

        
        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, FLAGS.confidence)
        
        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
        
    # Draw labels and boxes on the image
    detect = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels,height,labelh)
    
    return detect
