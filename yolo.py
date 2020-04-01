import numpy as np
import argparse
import cv2 as cv
import time
import os
from yolo_utils import infer_image, add_label
from PIL import Image
from pathlib import Path

FLAGS = []

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights',
                        type=str,
                        default='./yolov3-coco/yolov3hs.weights',
                        help='Path to the file which contains the weights for YOLOv3.')

    parser.add_argument('-cfg', '--config',
                        type=str,
                        default='./yolov3-coco/yolov3-custom.cfg',
                        help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-v', '--video-path',
                        type=str,
                        help='The path to the video file')

    parser.add_argument('-vo', '--video-output-path',
                        type=str,
                        default='./output.avi',
                        help='The path of the output video file')

    parser.add_argument('-l', '--labels',
                        type=str,
                        default='./yolov3-coco/coco-labels',
                        help='Path to the file having the labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence',
                        type=float,
                        default=0.5,
                        help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

    parser.add_argument('-th', '--threshold',
                        type=float,
                        default=0.3,
                        help='The threshold to use when applying the Non-Max Suppresion')

    parser.add_argument('-t', '--show-time',
                        type=bool,
                        default=False,
                        help='Show the time taken to infer each image.')
    parser.add_argument("-u", "--use_gpu", 
                        type=bool, 
                        default=False,
	                    help="boolean indicating if CUDA GPU should be used.")
    parser.add_argument("-d", "--display", 
                        type=bool, 
                        default=False,
	                    help="Show frame display")                                        

    FLAGS, unparsed = parser.parse_known_args()
    
    startt = time.time()
    # Load the weights and configutation to form the pretrained YOLOv3 model for smoking detection
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
    print("sucess")
   
    
    if FLAGS.use_gpu:
    	# set CUDA as the preferable backend and target
	    print("[INFO] setting preferable backend and target to CUDA...")
	    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
	    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

<<<<<<< HEAD
=======
    

>>>>>>> af60cff2a176e04ad0fa3d313ea08cd395870b98
    # Get the labels
    labels = open(FLAGS.labels).read().strip().split('\n')

    

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

<<<<<<< HEAD
=======
   

>>>>>>> af60cff2a176e04ad0fa3d313ea08cd395870b98
    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
<<<<<<< HEAD
    
=======
    print(layer_names)
>>>>>>> af60cff2a176e04ad0fa3d313ea08cd395870b98

    if FLAGS.video_path:
        
        #Extracting the audio file from the video file
        os.system('ffmpeg -i '+FLAGS.video_path+' -ab 160k -ac 2 -ar 44100 -vn Audio/'+Path(FLAGS.video_path).stem+'-audio.wav')
        
        vid = cv.VideoCapture(FLAGS.video_path)
        fps = vid.get(cv.CAP_PROP_FPS)
        fpsint = int(fps)
        print("FPS is :",fps)
        height , width =  None, None
        writer = None
        frameCount = 0
        
        while True:
            grabbed, frame = vid.read()
            print("Frame count",frameCount)

            # Checking if the complete video is read
            if not grabbed:
                break

            if width is None or height is None:
                height , width = frame.shape[:2]
            #Take first frame from each second for detection  

            if(frameCount%fpsint==0):
                frame, detect = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS, frameCount)

            # ims = cv.resize(frame, (960, 540))                    # Resize image
            # cv.imshow("Frame", ims)
            # key = cv.waitKey(1) & 0xFF    
            # if writer is None:
            #     # Initialize the video writer
            #     fourcc = cv.VideoWriter_fourcc(*"MJPG")
            #     writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, fps,(frame.shape[1], frame.shape[0]), True)
            # writer.write(frame)
                    #Check the frame contain any detection, if detection is occur, label statutory warning on the next 120 frames
            if(detect==2):
                for i in range(1,fpsint*5):
                    grabbed,frame = vid.read()
                    frameCount += 1
                    print("Frame count",frameCount)
                    height, width = frame.shape[:2]
                    add_label(frame,height,'smoke.png')
                    labelledImg = cv.imread("pasted_image.jpg")

                    if FLAGS.display:
                        ims = cv.resize(labelledImg, (960, 540))                    # Resize image
                        cv.imshow("Frame", ims)
                        key = cv.waitKey(1) & 0xFF
                    if writer is None:

                        # Initialize the video writer
                        fourcc = cv.VideoWriter_fourcc(*"MJPG")
                        writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, fps,(frame.shape[1], frame.shape[0]), True)
                    writer.write(labelledImg)
            elif(detect==2):
                for i in range(1,fpsint*5):
                    grabbed,frame = vid.read()
                    frameCount += 1
                    print("Frame count",frameCount)
                    height, width = frame.shape[:2]
                    add_label(frame,height,'helmet.png')
                    labelledImg = cv.imread("pasted_image.jpg")

                    if FLAGS.display:
                        ims = cv.resize(labelledImg, (960, 540))                    # Resize image
                        cv.imshow("Frame", ims)
                        key = cv.waitKey(1) & 0xFF
                    if writer is None:
                        # Initialize the video writer
                        fourcc = cv.VideoWriter_fourcc(*"MJPG")
                        writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, fps,
                                        (frame.shape[1], frame.shape[0]), True)
                        writer.write(labelledImg)        
            else:
                if FLAGS.display:

                    ims = cv.resize(frame, (960, 540))                    # Resize image
                    cv.imshow("Frame", ims)
                    key = cv.waitKey(1) & 0xFF            
                if writer is None:
                    # Initialize the video writer
                    fourcc = cv.VideoWriter_fourcc(*"MJPG")
                    writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, fps,
                                            (frame.shape[1], frame.shape[0]), True)
                writer.write(frame)
            frameCount += 1  
        print("[INFO] Cleaning up...")
        writer.release()
        vid.release()
        #Binding the audio file to the output.avi file
        os.system('ffmpeg -i output.avi -i Audio/'+Path(FLAGS.video_path).stem+'-audio.wav -c copy Video/'+Path(FLAGS.video_path).stem+'-Ouput.mkv')

        endt = time.time()

        print("The total time taken for entire process is :",endt-startt," Seconds")
    else:
        "Input video path is error"    
