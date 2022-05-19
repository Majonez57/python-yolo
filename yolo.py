from click import argument
import numpy as np
import argparse
import time
import cv2
import os

#Parse Arguments

aParser = argparse.ArgumentParser()
aParser.add_argument("-i", "--image", required=True, help="path to input image")
aParser.add_argument("-y","--yolo", required=True, help="base path to YOLO dic")
aParser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability of confidence")
aParser.add_argument("-t", "--threshold", type=float, default=0.3, help="Threshold for non-maxima suppression")
aParser.add_argument("-m", "--mini", type=bool, default=False, help="Whether to use Yolo-mini, or standard")
args = vars(aParser.parse_args())

#Load the coco and yolo stuff

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
labels = open(labelsPath).read().strip().split("\n")

if args["mini"]:
    print("== Loading YOLO-Tiny from file ==")
    weightPath = os.path.sep.join([args["yolo"], "yolov3-tiny.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3-tiny.cfg"])
else:
    print("== Loading YOLO from file ==")
    weightPath = os.path.sep.join([args["yolo"], "yolov3-spp.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3-spp.cfg"])


neuralNet = cv2.dnn.readNetFromDarknet(configPath, weightPath)

#Load input image, get its dimensions
image = cv2.imread(args["image"])
(H,W) = image.shape[:2]

#Grab output layer names from YOLO

layerNames = neuralNet.getLayerNames()
layerNames = [layerNames[i[0]-1] for i in neuralNet.getUnconnectedOutLayers()]

# Construct 'blob' from input image, pass through yolo
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True, crop=False)
neuralNet.setInput(blob)
start = time.time()
layerOut = neuralNet.forward(layerNames)
end = time.time()

#Show Timings
print("== YOLO took {:.6f}s to complete".format(end-start))

#Now, we want to visualise.

boxes = [] #all bounding boxes
confidences = [] #Confidence values that YOLO gives each object
classIDs = [] #DEtected object Class label

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

#TIme to draw said boxes

# Colours!
np.random.seed(94) 
colours = np.random.randint(0,255,size=(len(labels), 3), dtype="uint8")

if len(ind) > 0:
    for i in ind.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        colour = [int(c) for c in colours[classIDs[i]]]
        cv2.rectangle(image, (x,y), (x+w,y+h), colour, 2)
        text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x,y-5), cv2.FONT_HERSHEY_DUPLEX, 0.8, colour, 1)

#Show me the result!
cv2.imshow("Im", image)
cv2.waitKey(0)


