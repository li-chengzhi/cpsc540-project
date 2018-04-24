import numpy

def get_neighbour_indices4(i, m, n):
    (x,y) = numpy.unravel_index(i, (m,n))
    return (numpy.ravel_multi_index((a,b),(m,n)) for (a,b) in [(x-1, y), (x, y-1), (x+1, y), (x, y+1)] if -1<a<m and -1<b<n)

def get_neighbour_indices8(i, m, n):
    (x,y) = numpy.unravel_index(i, (m,n))
    return numpy.array([numpy.ravel_multi_index((a,b),(m,n)) for (a,b) in [(x-1,y-1), (x-1,y), (x-1,y+1), (x,y-1), (x,y), (x,y+1), (x+1,y-1), (x+1,y), (x+1,y+1)] if -1<a<m and -1<b<n])

def label_prop(X, y, threshold, m=None, n=None, neighbourhood_size = 8, verbose=False):
    if m==None and n==None:
        m = n = numpy.sqrt(X.shape[0]).astype(int)
    elif m==None:
        m = (X.shape[0]/n).astype(int)
    elif n==None:
        n = (X.shape[0]/m).astype(int)

    ''' Initialize the classifier and the neighbours of the labeled pixels '''
    labels = numpy.zeros(X.shape[0],dtype=bool)
    neighbourhood = set()

    if neighbourhood_size == 8:
        get_neighbour_indices = get_neighbour_indices8
    elif neighbourhood_size == 4:
        get_neighbour_indices = get_neighbour_indices4

    ''' Add neighbours off all labeled pixels (pre-training) to the
        neighbourhood '''
    labels[y==1] = True
    for i in numpy.nonzero(y)[0]:
        neighbourhood.update(get_neighbour_indices(i, m, n))

    while len(neighbourhood) > 0:
        i = neighbourhood.pop()
        if labels[i]:
            continue

        ''' Get the labeled neighbours '''
        neighbours = get_neighbour_indices(i, m, n)
        ii = neighbours[labels[neighbours]]

        ''' If the minimum difference between the pixel and the neighbouring
            labeled pixels is lower than the threshold, label the pixel
            and add its neighbours to the set of neighbours '''
        min_diff = numpy.min(numpy.linalg.norm(X[i] - X[ii], axis=1))
        
        if verbose:
            print(min_diff)
            
        if (min_diff < threshold):
            labels[i] = True
            for j in neighbours:
                if not labels[j]:
                    neighbourhood.add(j)

    return labels.astype(int)
