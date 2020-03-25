def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return (xB - xA) * (yB - yA)


x = [0,0,4,2]
y = [5,0,1+5,1+0]  
intersectionArea = bb_intersection_over_union(x,y)
unionArea = (x[2]*x[3])+(y[2]*y[3])-intersectionArea;
overlapArea = intersectionArea/unionArea; 
  
print("iou is :",overlapArea)