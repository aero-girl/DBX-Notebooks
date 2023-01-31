# Databricks notebook source
# MAGIC %pip install tesseract pytesseract opencv-python

# COMMAND ----------

import pytesseract #Tesseract Library
from pytesseract import Output
from PIL import Image

#Plotting Libraries
import matplotlib.pyplot as plt

# OpenCV Library
import cv2


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load image
# MAGIC To work with images in Spark OCR, we need to load them to a Spark DataFrame.

# COMMAND ----------

# file_name = spark.read.format("image").load("/FileStore/tables/Images-OCR/stampDS_00001.png")

# COMMAND ----------

imagePath = "/dbfs/FileStore/tables/Images-OCR/stampDS_00001.png"

# COMMAND ----------

image = Image.open(imagePath)
# image = image.resize((300,150))
image.save('sample.png')
image

# COMMAND ----------

# print(pytesseract.image_to_string(Image.open(imagePath)))


# COMMAND ----------

# binary_data_df = spark.read.format("binaryFile").load(imagePath)

# COMMAND ----------

img = cv2.imread(imagePath)
plt.imshow(img)

# COMMAND ----------

d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())

# COMMAND ----------

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = d["left"][i], d["top"][i], d["width"][i], d["height"][i] 
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) #Plotting bounding box
        img = cv2.putText(img, d['text'][i], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1) #Plotting texts on top of box
        
plt.imshow(img)

# COMMAND ----------

d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())

# COMMAND ----------


