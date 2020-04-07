import cv2 as cv


riding = ['motorcycling','riding a bike','riding scooter','riding mountain bike']
smoking = ['smoking','smoking hookah']
driving = ['driving car','driving tractor']
alcohol = ['tasting beer','drinking beer']
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
 
    
    
def detectChecking(img, boxes, confidences, classids, idxs, colors, labels,height,labelh):
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
                            x, y = boxes[j][0], boxes[j][1]
                            w, h = boxes[j][2], boxes[j][3]
                            helmetBox = [x,y,w,h]
                            iou = bb_intersection_over_union(motorBox,helmetBox)
                            
                            print("iou:",iou)
                            if iou < 0.25:
                                detect = True #person weared helmet
                                break
                        if detect: #if person not weared helmet return 1 for adding 
                            detect = 0
                        else:
                            detect = 1    
                                  
                    #if any without helmet class detects
                    if len(whelmet) != 0:
                        for j in whelmet:
                            x, y = boxes[i][0], boxes[i][1]
                            w, h = boxes[i][2], boxes[i][3]
                            whelmetBox = [x,y,w,h]
                            iou = bb_intersection_over_union(motorBox,whelmetBox)
                            print("iou :",iou)
                            if iou < 0.35:
                                detect = 1
                                return detect  #person not weared helmet
                                break
                return detect 
        #Smoking detection                  
        elif labelh in smoking:
            smoke = list(filter(lambda x: classids[x] == 3,idxs.flatten()))
            print(smoke)
            if len(smoke) > 0:
                detect = 2
                return detect
            else:
                detect = 0
                return detect
        elif labelh in driving:
            inside_car = list(filter(lambda x: classids[x] == 0,idxs.flatten()))
            woutseatbelt = list(filter(lambda x: classids[x] == 1,idxs.flatten()))
            wseatbelt = list(filter(lambda x: classids[x] == 2,idxs.flatten()))
            if len(inside_car) != 0:
                for i in inside_car:
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]
                    carBox = [x,y,w,h]
                    if len(wseatbelt) != 0:
                        for j in wseatbelt:
                            x, y = boxes[j][0], boxes[j][1]
                            w, h = boxes[j][2], boxes[j][3]
                            seatbeltBox = [x,y,w,h]
                            iou = bb_intersection_over_union(carBox,seatbeltBox)
                            
                            print("iou:",iou)
                            if iou < 0.25:
                                detect = True #person weared helmet
                                break
                        if detect: #if person not weared helmet return 1 for adding 
                            detect = 0
                        else:
                            detect = 3    
                                  
                    #if any without helmet class detects
                    if len(woutseatbelt) != 0:
                        for j in woutseatbelt:
                            x, y = boxes[j][0], boxes[j][1]
                            w, h = boxes[j][2], boxes[j][3]
                            wseatbeltBox = [x,y,w,h]
                            iou = bb_intersection_over_union(carBox,wseatbeltBox)
                            print("iou :",iou)
                            if iou < 0.25:
                                detect = 3
                                return detect  #person not weared helmet
                                break
                
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
