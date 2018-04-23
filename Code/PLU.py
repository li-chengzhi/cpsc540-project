from PIL import Image
import skimage.measure as skimage
import os

import numpy as np
from puAdapter import PUAdapter
from sklearn.ensemble import RandomForestClassifier


# Running options:
# ---------------------------------------------------
# Set to False to avoid displaying images when running
display_images = False

# Set to False to avoid saving images to the Tests folder
save_images = True


# Open image
# ---------------------------------------------------
data_dir = "../Dataset/"
image_name = "1.tif"
image = Image.open(data_dir + image_name)

# Convert image into ndarray
im_array = np.array(image)
im_height, im_width, n_chan = im_array.shape


# Extract subimage
# ---------------------------------------------------
n = 2000   # Roughly 1/7 of original height

# Extract upper right corner
dat = im_array[0:n,im_width-n:im_width,0:3]


# Parse manual labels
# http://nicodjimenez.github.io/boxLabel/annotate.html
# ---------------------------------------------------
# Read in file string
file_name = "1_1.txt"
raw_label_str = ""
with open(data_dir + file_name,"r") as f:
    for line in f:
        raw_label_str += line
f.close()

# Parse box boundaries containing labeled pixels from string
label_str = raw_label_str
labeled_pixels = []   # Array of pixel positions within the boxes
while True:
    beg = label_str.find("<code>")
    if beg == -1:   # End of string
        break
    end = label_str.find("</code>")
    bounds_str = label_str[beg+6:end]
    label_str = label_str[end+7:]
    bounds = [int(i) for i in bounds_str.split()]
    for x in range(bounds[1], bounds[3]+1):
        for y in range(bounds[0], bounds[2]+1):
            labeled_pixels.append([x,y])
labeled_pixels = np.asarray(labeled_pixels)

# Sample 1/k of the labeled pixels and pretend the rest is unlabeled
k = 500
labeled_sample = np.random.choice(labeled_pixels.shape[0],
                                  int(np.floor(labeled_pixels.shape[0]/k)),
                                  replace=False)
labeled_sample = labeled_pixels[labeled_sample,:]

# Create nxn label matrix with 1 being labeled and -1 being unlabeled
dat_labels = -np.ones((n,n))
def temp(x):
    dat_labels[x[0],x[1]] = 1
np.apply_along_axis(temp, 1, labeled_sample)


# Reduce pixels in subimage for computational efficiency
# ---------------------------------------------------
# Reduce total number of pixels by (1/s)^2 (with max pooling)
s = 10
dat = skimage.block_reduce(dat, (s,s,1), np.max)
dat_labels = skimage.block_reduce(dat_labels, (s,s), np.max)

# Get list of labeled pixel positions of reduced image
dat_labels = list(zip(*np.where(dat_labels==1)))

# Compute size of reduced image
n = int(np.floor(n/s))

# Save subimage
test_dir = "../Tests/"
image_num = "1"   # Change when running on different subimage
im_sub = Image.fromarray(dat)
if save_images:
    save_name = test_dir + "subimage" + image_num + ".png"
    try:
        os.remove(save_name)
    except OSError:
        pass
    im_sub.save(save_name)


# Display sampled labeled pixels on subimage
# ---------------------------------------------------    
dat_labeled = np.copy(dat)

# Colour the labeled pixels
def colour_labeled(x):
    dat_labeled[x[0],x[1]] = [255,0,0]   # Red
    
np.apply_along_axis(colour_labeled, 1, dat_labels)

im_labeled = Image.fromarray(dat_labeled)
if save_images:
    save_name = test_dir + "labeled" + image_num + ".png"
    try:
        os.remove(save_name)
    except OSError:
        pass
    im_labeled.save(save_name)
if display_images:
    im_labeled.show()


# PUL 1 - Random forest with only pixel RGB values
# https://github.com/aldro61/pu-learning/blob/master/src/examples/puAdapterExample.py
# ---------------------------------------------------
# Prepare training set
X = np.reshape(dat, (n*n,3))
y = -np.ones(n*n)

# Convert (x,y) index to a flattened 1D-array index
def get_flattened_index(x,y,n):
    i, = np.unravel_index(np.ravel_multi_index((x,y), (n,n)), n*n)
    return i

# Set the corresponding labeled pixel in y to 1
def set_labeled(x):
    y[get_flattened_index(x[0],x[1],n)] = 1
    
np.apply_along_axis(set_labeled, 1, dat_labels)

# Fit PUL random forest classifier
estimator = RandomForestClassifier(n_estimators=500,
                                   criterion='gini',
                                   bootstrap=True)
pu_estimator = PUAdapter(estimator)
pu_estimator.fit(X,y)

# Get the fitted training labels
yhat = pu_estimator.predict(X)
yhat = np.asarray(yhat)

# Convert the 1D-array indices into (x,y) indices
yhat_ii = [np.unravel_index(i, (n,n)) for i,x in enumerate(yhat) if x==1]

# Display predicted labels
dat_labeled = np.copy(dat)
np.apply_along_axis(colour_labeled, 1, yhat_ii)

im_pul1 = Image.fromarray(dat_labeled)
if save_images:
    save_name = test_dir + "pul1_" + image_num + ".png"
    try:
        os.remove(save_name)
    except OSError:
        pass
    im_pul1.save(save_name)
if display_images:
    im_pul1.show()


# PUL 2 - Random forest with pixel and 8-neighbour RGB values
# Note: for convenience, use only pixels with 8-neighbours
#       so resulting image will be (n-2)x(n-2)
# ---------------------------------------------------
# Prepare training set
X = []
for x in range(1,n-1):
    for y in range(1,n-1):
        row = []
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
X = np.asarray(X)
y = -np.ones((n-2)**2)

# Set the corresponding labeled pixel in y to 1
def set_labeled(x):
    if (x[0] != 0 and x[0] != n-1) and x[1] != 0 and x[1] != n-1:
        y[get_flattened_index(x[0]-1,x[1]-1,n-2)] = 1
    
np.apply_along_axis(set_labeled, 1, dat_labels)

# Fit PUL random forest classifier
estimator = RandomForestClassifier(n_estimators=500,
                                   criterion='gini',
                                   bootstrap=True)
pu_estimator = PUAdapter(estimator)
pu_estimator.fit(X,y)

# Get the fitted training labels
yhat = pu_estimator.predict(X)
yhat = np.asarray(yhat)

# Convert the 1D-array indices into (x,y) indices
yhat_ii = [np.unravel_index(i, (n-2,n-2)) for i,x in enumerate(yhat) if x==1]

# Display predicted labels
dat_labeled = np.copy(dat[1:n-1,1:n-1])
np.apply_along_axis(colour_labeled, 1, yhat_ii)

im_pul2 = Image.fromarray(dat_labeled)
if save_images:
    save_name = test_dir + "pul2_" + image_num + ".png"
    try:
        os.remove(save_name)
    except OSError:
        pass
    im_pul2.save(save_name)
if display_images:
    im_pul2.show()
