from PIL import Image
import numpy
import skimage.measure as skimage
from scipy.cluster.vq import kmeans, vq

# load image
im = Image.open("test.tif")
# im.show()

# convert image into ndarray
imarray = numpy.array(im)

# dat = numpy.reshape(imarray, (7709*6001, 4))

# work with subimage
n = 2000 # roughly 1/7 of height
width = imarray.shape[1]
dat = imarray[0:n,width-n:width,:]

# reduce number pixels by (1/2)^2 (max pooling)
dat = skimage.block_reduce(dat, (2,2,1), numpy.max)
n = 1000

#n = 256
#dat = imarray[0:n,:,:]
#d = dat.shape[1]

# display subimage
im2 = Image.fromarray(dat)
im2.show()

# reshape into 2d-array
dat2 = numpy.reshape(dat, (n*n, 4))

# convert into float
dat3 = dat2 / 255

# kmeans -- returns list of means ("codebook")
codebook, distortion = kmeans(dat3, 3, 100)

# vector quant -- returns index of closest mean for each pixel ("code")
code, dist = vq(dat3, codebook)    

# replace list of index with mean
dat4 = numpy.asarray([codebook[i] for i in code])

# convert back into uint8 and reshape back into subimage
dat4 = dat4 * 255
dat4 = dat4.round()
dat4 = numpy.uint8(dat4)
dat5 = numpy.reshape(dat4, (n,n,4))

# display vector quant image
res = Image.fromarray(dat5)
res.show()