import cv2
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os , ssl ,time
from PIL import Image
import PIL.ImageOps

if(not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context',None)):
    ssl.create_default_https_context = ssl._create_unverified_context

x , y = fetch_openml('mnist_784' , version = 1 , return_X_y = True)
#print(pd.Series(y).value_counts())

classes = [ 0 ,1 ,2 ,3 , 4, 5 , 6 , 7 , 8 , 9]
nclasses = len(classes)

x_train , x_test , y_train , y_test = train_test_split(x,y , random_state = 9 , train_size = 7500 , test_size = 2500)

samples_per_class = 5
figure = plt.figure(figsize = (nclasses*2 , (1 + samples_per_class*2)))

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(x_train_scaled , y_train)

def get_prediction(image):
    im_pil = Image.fromarray(image)

    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28 , 28) , Image.ANTIALIAS)

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20
    max_pixel = np.percentile(image_bw_resized_inverted , pixel_filter)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]

