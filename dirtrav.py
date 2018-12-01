import os
import sys
import numpy as np
from PIL import Image
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

path = "C:/raj/imag"

i = 0

print path

for dirname , dirnames , filenames in os.walk ( path ):
        for file_ in filenames :
            naam=os.path.join(dirname, file_);
            im = cv2.imread(naam)
            faces = face_cascade.detectMultiScale(im, 1.3, 5)
            
            for(x,y,w,h) in faces:
                img = cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
                roi_color = img[y:y+h, x:x+w]
                name=dirname+str(1)+"\\"+file_;
                print ('Creating...' + name)
                cv2.imwrite(name, roi_color)
                i=i+1;
	    
print('no of files are', i)