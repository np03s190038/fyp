#!/usr/bin/env python
# coding: utf-8

# In[3]:Importing main libraries

import cv2
import matplotlib.pyplot as plt

# In[4]: Loading config file and model

config_file = 'object_detection_model/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt'
frozen_model = 'object_detection_model/frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# In[15]: Loading class Labels for various objects

classLabels = []
file_name = 'dataset/labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# In[16]: Setting the default size for the model

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# In[19]: read an image
img = cv2.imread('input_images/im1jpg')

# In[41]:
ClassIndex, Confidence, bbox = model.detect(img, confThreshold=0.5)

# In[44]: Product Labeling and box boundary for the detecting image
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), Confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

# In[45]: displaying detected image

cv2.imshow("Output",img)
cv2.waitKey(0)







