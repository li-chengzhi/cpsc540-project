from PIL import Image
import numpy
import skimage.measure as skimage
from sklearn.semi_supervised import label_propagation

# load image
im = Image.open("clouds.tif")

# convert image into ndarray
imarray = numpy.array(im)

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
labeled_sample = numpy.random.choice(labeled.shape[0], int(numpy.floor(labeled.shape[0]/1000)), replace=False)
labeled_sample = labeled[labeled_sample,:]


# reduce number pixels by (1/s)^2 (max pooling)
s = 20
labeled_2d = numpy.zeros((n,n))
def labels(pixel):
    labeled_2d[pixel[0],pixel[1]] = 1
numpy.apply_along_axis(labels, 1, labeled_sample)

dat = skimage.block_reduce(dat, (s,s,1), numpy.max)
labeled_2d = skimage.block_reduce(labeled_2d, (s,s), numpy.max)
lab_samp_reduced = list(zip(*numpy.where(labeled_2d==1)))
n = int(numpy.floor(n/s))

dat_labeled = dat

def colour_labeled(pixel):
    dat_labeled[pixel[0],pixel[1]] = [255,192,203,0]
    
numpy.apply_along_axis(colour_labeled, 1, lab_samp_reduced)

im2 = Image.fromarray(dat_labeled)
im2.show()

X = numpy.reshape(dat, (n*n,4))
y = numpy.reshape(labeled_2d, (n*n))

label_prop = label_propagation.LabelPropagation(kernel='knn')
label_prop.fit(X,y)

trans_labels = label_prop.transduction_
trans_lab_2d = numpy.asarray(trans_labels)
trans_lab_2d = numpy.reshape(trans_lab_2d, (n,n))

#dat_labeled = dat
#numpy.apply_along_axis(colour_labeled, 1, lab_samp_reduced)
#
#im2 = Image.fromarray(dat_labeled)
#im2.show()