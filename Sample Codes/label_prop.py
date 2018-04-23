import numpy

def distance(x1, x2):
    return numpy.linalg.norm(x2-x1)

def neighbors(x, y):
    return np.array()

def get_neighbour_indices4(i, m, n):
    (x,y) = numpy.unravel_index(i, (m,n))
    return (numpy.ravel_multi_index((a,b),(m,n)) for (a,b) in [(x-1, y), (x, y-1), (x+1, y), (x, y+1)] if a>=0 and a<m and b>=0 and b<n)


def desire(X,mask,i,m,n):
    energy = 1000
    for j in get_neighbour_indices4(i, m, n):
        if mask[j]:
            energy = min(energy,distance(X[i], X[j]))
    return energy


def label_prop(X, y, m, n, threshold):
    mask = numpy.zeros(X.shape[0],dtype=bool) # the classifier
    neighbourhood = set() # ther current neighbourhood of the labeled pixels

    print(X[0])
    # Initialize
    mask[y==1] = True
    for i in numpy.nonzero(y)[0]:
        neighbourhood.update(get_neighbour_indices4(i, m, n))

    while len(neighbourhood) > 0:
        #print(len(neighbourhood))
        i = neighbourhood.pop()
        if mask[i]:
            continue
        neighbours = get_neighbour_indices4(i, m, n)
        if (desire(X,mask,i,m,n) < threshold):
            mask[i] = True
            for j in neighbours:
                if not mask[j]:
                    neighbourhood.add(j)

    return mask.astype(int)
