from PIL import Image
import numpy
import skimage.measure as skimage
from sklearn.semi_supervised import label_propagation

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

# display sampled labeled pixels
dat_labeled = numpy.copy(dat)

def colour_labeled(pixel):
    dat_labeled[pixel[0],pixel[1]] = [255,0,0]
    
numpy.apply_along_axis(colour_labeled, 1, lab_samp_reduced)

im2 = Image.fromarray(dat_labeled)
im2.show()


# label propagation inputs
X = numpy.reshape(dat, (n*n,3))
y = numpy.reshape(labeled_2d, (n*n))

# set -1 for unlabeled
y[y == 0] = -1

# set weight based on RGB values
def colour_threshold(x):
#    return numpy.linalg.norm(x) < 50
    return numpy.exp(-numpy.linalg.norm(x))
#    return 1

# convert (x,y) index into a flattened 1d array index
def get_flattened_index(x,y,n1,n2):
    i, = numpy.unravel_index(numpy.ravel_multi_index((x,y), (n1,n1)), n2)
    return i

# include a negative label...
y[get_flattened_index(n-1,0,n,n*n)] = 0

# custom kernel for label propagation
# only 8 pixel neighbours may be non-zero
def custom_kernel(X1, X2):
    n1, d1 = X1.shape
    n = int(numpy.sqrt(n1))
    W = numpy.zeros((n1,n1))
    for i in range(n1):
        x,y = numpy.unravel_index(i, (n,n))
        W[i,i] = 1
        if y < n-1:
            ii = get_flattened_index(x,y+1,n,n1)
            W[i,ii] = W[ii,i] = colour_threshold(X1[i,:]-X2[ii,:])
            if x > 0:
                ii = get_flattened_index(x-1,y+1,n,n1)
                W[i,ii] = W[ii,i] = colour_threshold(X1[i,:]-X2[ii,:])
            if x < n-1:
                ii = get_flattened_index(x+1,y+1,n,n1)
                W[i,ii] = W[ii,i] = colour_threshold(X1[i,:]-X2[ii,:])
        if x < n-1:
            ii = get_flattened_index(x+1,y,n,n1)
            W[i,ii] = W[ii,i] = colour_threshold(X1[i,:]-X2[ii,:])
            if y > 0:
                ii = get_flattened_index(x+1,y-1,n,n1)
                W[i,ii] = W[ii,i] = colour_threshold(X1[i,:]-X2[ii,:])
    return W
#    return -numpy.ones((X1.shape[0], X1.shape[0]))

label_prop = label_propagation.LabelPropagation(custom_kernel, max_iter=50, n_jobs=-1)
label_prop.fit(X,y)

trans_labels = label_prop.transduction_
trans_labels = numpy.asarray(trans_labels)
#trans_lab_2d = numpy.reshape(trans_lab_2d, (n,n))

trans_labs = [numpy.unravel_index(i, (n,n)) for i, x in enumerate(trans_labels) if x == 1]

# display resulting labels
dat_labeled = numpy.copy(dat)
numpy.apply_along_axis(colour_labeled, 1, trans_labs)

im2 = Image.fromarray(dat_labeled)
im2.show()
