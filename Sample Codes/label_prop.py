import numpy
from sets import Set

def distance(x1, x2):
    return numpy.linalg.norm(x2-x1)

def get_neighbour_indices8(i, m, n):
    '''
    gets the index of the neighbours of a pixel i in an mxn image
    needs to make sure that it doesn't include pixels outside the image
    example:
        for i=0, m=3, n=4 we should get (1, 3)
    '''

def label_prop(X, y, m, n):
    mask = numpy.zeros((m,n),dtype=bool)
    neighbourhood = numpy.
