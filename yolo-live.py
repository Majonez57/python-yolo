from fractions import Fraction
from xmlrpc.client import Boolean
from click import argument
import numpy as np
import argparse
import time
import cv2
import glob
import os

#Parse Arguments

aParser = argparse.ArgumentParser()
aParser.add_argument("-y","--yolo", required=True, help="base path to YOLO dic")
aParser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability of confidence")
aParser.add_argument("-t", "--threshold", type=float, default=0.3, help="Threshold for non-maxima suppression")
aParser.add_argument("-m", "--mini", type=Boolean, default=False, help="Whether to use Yolo-mini, or standard")
args = vars(aParser.parse_args())

#Load the coco and yolo stuff

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
labels = open(labelsPath).read().strip().split("\n")

if args["mini"]:
    weightPath = os.path.sep.join([args["yolo"], "yolov3-tiny.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3-tiny.cfg"])
else:
    weightPath = os.path.sep.join([args["yolo"], "yolov3-spp.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3-spp.cfg"])
print("== Loading YOLO from file ==")
neuralNet = cv2.dnn.readNetFromDarknet(configPath, weightPath)


def singleFrame(layerOut, image):
    boxes = [] #all bounding boxes
    confidences = [] #Confidence values that YOLO gives each object
    classIDs = [] #DEtected object Class label
    (H,W) = image.shape[:2]

    for output in layerOut:
        for detect in output:
            #get ID and confidence
            scores = detect[5:]
            classID = np.argmax(scores)
            conf = scores[classID]

            #Filter weak predictions
            if conf > args["confidence"]:
                #scale boxes
                box = detect[0:4] * np.array([W, H, W, H])
                (cenX, cenY, w, h) = box.astype("int")

                x = int(cenX - (w / 2))
                y = int(cenY - (h / 2))

                boxes.append([x,y,int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)
    
    #Apply the NMS (Get rid of overlaps)
    ind = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    #time to draw said boxes

    # Colours!
    np.random.seed(42) 
    colours = np.random.randint(0,250,size=(len(labels), 3), dtype="uint8")

    if len(ind) > 0:
        for i in ind.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            colour = [int(c) for c in colours[classIDs[i]]]
            cv2.rectangle(image, (x,y), (x+w,y+h), colour, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x,y-5), cv2.FONT_HERSHEY_DUPLEX, 1.2, colour, 1)

video = cv2.VideoCapture(0)

while True:
    start = time.time()
    b, frame = video.read()

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), [0,0,0], 1, crop=False)
    neuralNet.setInput(blob)

    layerNames = neuralNet.getLayerNames()
    layerNames = [layerNames[i[0]-1] for i in neuralNet.getUnconnectedOutLayers()]

    lO = neuralNet.forward(layerNames)
    singleFrame(lO, frame)
    cv2.putText(frame, "FPS: " + str(int(1.0/ (time.time() - start))), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
    cv2.imshow('live', frame)
    cv2.waitKey(1)
