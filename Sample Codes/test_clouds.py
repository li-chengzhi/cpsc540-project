from PIL import Image
import numpy
import random

# load image
im = Image.open("clouds.tif")
# im.show()

# convert image into ndarray
imarray = numpy.array(im)

# dat = numpy.reshape(imarray, (7709*6001, 4))

# work with subimage
n = 2000 # roughly 1/7 of height
width = imarray.shape[1]
dat = imarray[0:n,width-n:width,:]

# http://nicodjimenez.github.io/boxLabel/annotate.html
file_labels = ""
with open("clouds.txt","r") as f:
    for line in f:
        file_labels += line

str_labels = file_labels
labeled = []
while True:
    beg = str_labels.find("<code>")
    if beg == -1:
        break
    end = str_labels.find("</code>")
    str_bounds = str_labels[beg+6:end]
    str_labels = str_labels[end+7:]
    bounds = [int(i) for i in str_bounds.split()]
    for x in range(bounds[1], bounds[3]):
        for y in range(bounds[0], bounds[2]):
            labeled.append([x,y])

labeled = numpy.asarray(labeled)
labeled_sample = numpy.random.choice(labeled.shape[0], int(numpy.floor(labeled.shape[0]/4)), replace=False)
labeled_sample = labeled[labeled_sample,:]

dat_labeled = dat

def colour_labeled(pixel):
    dat_labeled[pixel[0],pixel[1]] = [255,192,203,0]
    
numpy.apply_along_axis(colour_labeled, 1, labeled_sample)

im2 = Image.fromarray(dat_labeled)
im2.show()

X = []
for x in range(1,n-2):
    for y in range(1,n-2):
        # features:
        # (x, y, <pixel values>, <8 neighbour pixel values>)
        row = [x,y]
        row += dat[x,y,:].tolist()
        row += dat[x-1,y-1,:].tolist()
        row += dat[x,y-1,:].tolist()
        row += dat[x+1,y-1,:].tolist()
        row += dat[x-1,y,:].tolist()
        row += dat[x+1,y,:].tolist()
        row += dat[x-1,y+1,:].tolist()
        row += dat[x,y+1,:].tolist()
        row += dat[x+1,y+1,:].tolist()
        X.append(row)
X = numpy.asarray(X)
Y = numpy.zeros((n,n))

def set_label(pixel):
    Y[pixel[0],pixel[1]] = 1

numpy.apply_along_axis(set_label, 1, labeled_sample)
Y = Y[1:n-2,1:n-2]
Y = numpy.reshape(Y,((n-3)*(n-3),1))

# https://github.com/phuijse/bagging_pu/blob/master/PU_Learning_simple_example.ipynb
from sklearn.tree import DecisionTreeClassifier

NP = X.shape[0]
NU = n*n - NP

T = 10
K = NP
train_label = numpy.zeros(NP+K)
train_label[:NP] = 1.0
