from __future__ import division
import os
import sys
from PIL import Image
import xml.etree.ElementTree as ET

stride = 50
resize = 224


xmin=[]
ymin=[]

def resize(folder, fileName):
    print(fileName)
    global xmin
    global ymin
    xmin=[]
    ymin=[]
    filePath = os.path.join(folder, fileName)
    im = Image.open(filePath)
    pix = im.load()
    w, h  = im.size
    
    xmlPath = os.path.join('/media/bashir/10e69c97-5ae7-4ec6-890d-6ab57ba4f1bd/bashir/Plosmodium/plasmodium-images/annotation', fileName[:-4] +'.xml')
    #print(xmlPath)
    e = ET.parse(xmlPath).getroot()
    
    
    #/home/bashir/Plosmodium/plasmodium-images/annotation/plasmodium-2528.xml
    for a in e.iter('xmin'):
        xmin.append(int(float(a.text)))
        #print(a.text)
    for a in e.iter('ymin'):
        ymin.append(int(float(a.text)))
    
    assert len(xmin) == len(ymin)
    
    count = 0

    for i in xrange(len(xmin)):
        newIm1 = im.crop((xmin[i],   ymin[i],   xmin[i]+stride, ymin[i]+stride)).resize((224, 224),Image.ANTIALIAS)
        newIm1.save(os.path.join("/media/bashir/10e69c97-5ae7-4ec6-890d-6ab57ba4f1bd/bashir/Plosmodium/training224/malaria", fileName[:-4] +"_"+ str(i)+".jpg"))
    
    
    for x in range(0,w-stride,stride):
        for y in range(0,h-stride,stride):
            if isNormal(x,y):
                newIm1 = im.crop((x,   y,   x+stride, y+stride)).resize((224, 224),Image.ANTIALIAS)
                p = os.path.join("/media/bashir/10e69c97-5ae7-4ec6-890d-6ab57ba4f1bd/bashir/Plosmodium/training224/normal", fileName[:-4] +"_"+ str(x)+"_"+str(y)+".jpg")
                newIm1.save(p)
		count +=1
		if count >= len(xmin):
			return
                #print(p)
            
    
        


def isNormal(w,h):
    #print('xmin size',len(xmin))
    for i in xrange(len(xmin)):
        #print("checking",w,xmin[i],h,ymin[i])
        if abs(w-xmin[i])<stride and abs(h-ymin[i])<stride:
            #print("Failed",w,xmin[i],h,ymin[i])
            return False
    #print("Passed",w,h)
    return True
          
            

def bulkResize(imageFolder):
    imgExts = ["png", "bmp", "jpg"]
    gc =0
    for path, dirs, files in os.walk(imageFolder):
	
        for fileName in files:
	    #print (fileName[:-4])
            ext = fileName[-3:].lower()
            if ext not in imgExts:
                continue
            
            resize(path, fileName)
            
            
            
            

if __name__ == "__main__":
    imageFolder=sys.argv[1] # first arg is path to image folder
    #resizeFactor=float(sys.argv[2])/100.0# 2nd is resize in %
    bulkResize(imageFolder)
