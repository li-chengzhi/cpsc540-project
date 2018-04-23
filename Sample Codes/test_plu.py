from PIL import Image
import numpy
import skimage.measure as skimage
from puAdapter import PUAdapter
from sklearn.ensemble import RandomForestClassifier

# load image
im = Image.open("clouds.tif")

# convert image into ndarray
imarray = numpy.array(im)

# work with subimage
n = 2000 # roughly 1/7 of original height
width = imarray.shape[1]
dat = imarray[0:n,width-n:width,0:3]

# http://nicodjimenez.github.io/boxLabel/annotate.html
# get manually labeled cloud pixels
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
    for x in range(bounds[1], bounds[3]+1):
        for y in range(bounds[0], bounds[2]+1):
            labeled.append([x,y])

labeled = numpy.asarray(labeled)
labeled_sample = numpy.random.choice(labeled.shape[0], int(numpy.floor(labeled.shape[0]/500)), replace=False)
labeled_sample = labeled[labeled_sample,:]


# reduce number pixels by (1/s)^2 (max pooling)
s = 10
labeled_2d = numpy.zeros((n,n))
def labels(pixel):
    labeled_2d[pixel[0],pixel[1]] = 1
numpy.apply_along_axis(labels, 1, labeled_sample)

dat = skimage.block_reduce(dat, (s,s,1), numpy.max)
labeled_2d = skimage.block_reduce(labeled_2d, (s,s), numpy.max)
lab_samp_reduced = list(zip(*numpy.where(labeled_2d==1)))
n = int(numpy.floor(n/s))


# display sampled labeled pixels
dat_labeled = numpy.copy(dat)

def colour_labeled(pixel):
    dat_labeled[pixel[0],pixel[1]] = [255,0,0]
    
numpy.apply_along_axis(colour_labeled, 1, lab_samp_reduced)

im2 = Image.fromarray(dat_labeled)
im2.show()

# -------------------------
# PUL - Random Forest with single pixel value features
X = numpy.reshape(dat, (n*n,3))

y = -numpy.ones(n*n)

# convert (x,y) index into a flattened 1d array index
def get_flattened_index(x,y,n1,n2):
    i, = numpy.unravel_index(numpy.ravel_multi_index((x,y), (n1,n1)), n2)
    return i

def set_label(pixel):
    y[get_flattened_index(pixel[0],pixel[1],n,n*n)] = 1
    
numpy.apply_along_axis(set_label, 1, lab_samp_reduced)

estimator = RandomForestClassifier(n_estimators=500,
                                   criterion='gini',
                                   bootstrap=True)
pu_estimator = PUAdapter(estimator)

pu_estimator.fit(X,y)

y_fit = pu_estimator.predict(X)
y_fit = numpy.asarray(y_fit)

y_fit_pos = [numpy.unravel_index(i, (n,n)) for i, x in enumerate(y_fit) if x == 1]

# display resulting labels
dat_labeled = numpy.copy(dat)
numpy.apply_along_axis(colour_labeled, 1, y_fit_pos)

# -------------------------------
# PUL - Random forest with single and all 8 neighbour pixel value features

#X = []
#for x in range(1,n-1):
#    for y in range(1,n-1):
#        # features:
#        # (<pixel values>, <8 neighbour pixel values>)
#        row = []
#        row += dat[x,y,:].tolist()
#        row += dat[x-1,y-1,:].tolist()
#        row += dat[x,y-1,:].tolist()
#        row += dat[x+1,y-1,:].tolist()
#        row += dat[x-1,y,:].tolist()
#        row += dat[x+1,y,:].tolist()
#        row += dat[x-1,y+1,:].tolist()
#        row += dat[x,y+1,:].tolist()
#        row += dat[x+1,y+1,:].tolist()
#        X.append(row)
#X = numpy.asarray(X)
#
#y = -numpy.ones((n-2)**2)
#
## convert (x,y) index into a flattened 1d array index
#def get_flattened_index(x,y,n1,n2):
#    i, = numpy.unravel_index(numpy.ravel_multi_index((x,y), (n1,n1)), n2)
#    return i
#
#def set_label(pixel):
#    if pixel[0] != 0 & pixel[0] != n-1 & pixel[1] != 0 & pixel[1] != n-1:
#        y[get_flattened_index(pixel[0]-1,pixel[1]-1,n-2,(n-2)**2)] = 1
#    
#numpy.apply_along_axis(set_label, 1, lab_samp_reduced)
#
## https://github.com/aldro61/pu-learning/blob/master/src/examples/puAdapterExample.py
#
#estimator = RandomForestClassifier(n_estimators=500,
#                                   criterion='gini',
#                                   bootstrap=True)
#pu_estimator = PUAdapter(estimator)
#
#pu_estimator.fit(X,y)
#
#y_fit = pu_estimator.predict(X)
#y_fit = numpy.asarray(y_fit)
#
#y_fit_pos = [numpy.unravel_index(i, (n-2,n-2)) for i, x in enumerate(y_fit) if x == 1]
#
## display resulting labels
#dat_labeled = numpy.copy(dat[1:n-1,1:n-1])
#numpy.apply_along_axis(colour_labeled, 1, y_fit_pos)
# -------------------------

im2 = Image.fromarray(dat_labeled)
im2.show()

