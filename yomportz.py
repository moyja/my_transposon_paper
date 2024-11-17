# @author = moyja

########################################################################################################################

'''

Notes to self:
    1.  you awlays dot product sparse.dot(full) the other way does a weird matrix of sparses thing.
    2. for a 6 x 6 figure, font size 10 is appropriate with 13 for the title
    3. It seems that lil_array indexing (setting elements randomly on the fly) is about 6x slower than dense
    4. dok array i want to say appears to take 3x longer to set elements than lil_array so idk what the fucking point is, its definitely some 2x slower for dot product too, seems like dok array is just absolute shite. i think csr array and lil array are the only one worth a dman.
    5. okay call me crazy but it seems like for dot product lil array is just as good as csr array
    6. okay so interesting. for lil_array.dot(dense square) does just as good as csr array, but when multiplying by single vector csr is much much faster. but converting to csr format takes just as long. it appears that the way lil matrix works is that it first converst to csr format, and when the computation itself is long, in the case of a matrix matrix multiplication, the conversion takes only a itny bit of the time, so it is irrelevant, but for matrix vector its actually the majority of the time.

Table of contents
    # 1: imports
    # 3: itertools
    # 4: my guys
    # 5: catz
    # 6: funky fucks
    #### gpt code

'''

########################################################################################################################

# 1: imports

# first my local imports
from sexy_prog_bar import *

# then the big dic imports
import contextlib
import collections
import colorama
import copy
import gc
import glob
import itertools as it
import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nra
import numpy.linalg as nla
import os
import pandas as pd
import pickle as pk
#import pynverse
import re
import scipy as sp
import scipy.linalg as sla
import scipy.sparse as sps
import sys
import time

from collections import Counter, defaultdict
#from colorama import Fore, Style
from copy import deepcopy
from heapq import heapify, heappop, heappush, heapreplace

from numpy import ( angle, arange, arccos, array, arcsin, arctan, arctan2, arctanh, argsort,
                   clip, concatenate, conj, cos, cosh, cumsum, diag, diagonal,
                   einsum, exp, eye, fill_diagonal, flip, floor, imag, ix_, kron,
                   linspace, log, logical_and, logical_not, logical_or, logical_xor, logspace,
                   mean, median, meshgrid, nan_to_num, ones, pi, quantile, real, repeat, rint,
                   sign, sin, sinc, sinh, sqrt, std, swapaxes,
                   tan, tanh, tile, trace, unique, vectorize, where, zeros )
from numpy.linalg import norm
from numpy.random import binomial, poisson, rand, randint, randn

#from pynverse import inversefunc

from queue import PriorityQueue

from scipy.linalg import null_space, orth, sqrtm
from scipy.sparse import csgraph, csr_array, diags, eye as speye, kron as sparkron, lil_array
from scipy.special import comb, erf, erfinv, expit, factorial, logsumexp, lambertw
from scipy.stats import entropy, skew, kurtosis
from time import perf_counter

'''

# other imports ive made use of before

import cvxpy as cp
import networkx as nx
import sklearn as sk

from mne_connectivity.viz import plot_connectivity_circle
from pySankey.sankey import sankey
from networkx.algorithms.isomorphism import rooted_tree_isomorphism
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from sklearn.linear_model import LogisticRegression

'''

########################################################################################################################

########################################################################################################################

# itertools

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))

################################################################################################################################

# 4: my guys

class my_bag(collections.Counter):
    def __hash__(self):
        return hash(tuple(sorted(self.elements())))
    
    def __iter__(self):
        return self.elements()
    
    def total(self):
        return sum(list(self.values()))


class my_csr(sps.csr_array):
    # a csr array with little padding bits so that you can efficiently add boiz to it
    # requires a final row be appended to the thing its attacking to show up in the dot product with our padding.
    # outputs a guy with an extra row as well
    
    # what if instead of this we break every matrix mult into two guys, M and dM
    # the dM is a lil matrix which doesnt take much time to run since it's small,
    # the M is a csr which is large and fast.
    def __init__( self, arg1, shape = None, dtype = None, pad = (10, 1) ):
        # pad = pad[0] + pad[2] * L
        # havent quite checked to ensure no nonzero values yet
        og_array = sps.csr_array(arg1, shape=None, dtype=None)
        
        data = og_array.data
        indices = og_array.indices
        indptr = og_array.indptr
        shape = og_array.shape
        M = np.prod(shape)
        
        lengths = indptr[1:] - indptr[:-1]
        padding = pad[0] + (pad[1] * lengths).astype(int)
        new_lengths = lengths + padding
        L = np.sum(new_lengths)
        
        new_data = zeros(L)
        new_indices = zeros(L)
        new_indptr = cat(zeros(1, dtype = int), cumsum(new_lengths), array([L]))
        new_shape = (shape[0] + 1, shape[1] + 1)
        
        for i, j, k, l in zip(indptr[:-1], new_indptr[:-2], padding, lengths):
            new_data[ j : j + l ] = data[i : i + l]
            new_data[ j + l : j + l + k ] = zeros(k)
            
            new_indices[ j : j + l ] = indices[i : i + l]
            new_indices[ j + l : j + l + k ] = shape[1]
        
        super().__init__( (new_data, new_indices, new_indptr), shape = new_shape)
    
    
class hash_dict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))
    
    
class hash_set(set):
    def __hash__(self):
        return hash(tuple(sorted(self)))


class my_pq:
    
    def __init__(self, stuff = [], priorities = []):
        self.pq = PriorityQueue()
        if len(stuff) != len(priorities):
            raise Exception('shape imsmatch')
        for thing, priority in zip(stuff, priorities):
            self.pq.put(( -priority, thing ))
    
    ######## iterator shit in progress,
    
    def __iter__(self):
        self.iter = deepcopy(self.pq)
        return deepcopy(self)
    
    def __next__(self):
        if self.iter.empty():
            raise StopIteration
        else:
            return self.iter.get()
    
    ######## iterator shit in progress,
    
    def empty(self):
        return self.pq.empty()
            
    def get(self, return_priority = False):
        if self.pq.empty():
            raise Exception('queue empty')
        else:
            priority, thing = self.pq.get()
            if return_priority:
                return thing, -priority
            else:
                return thing
    
    def put(self, thing, priority):
        self.pq.put(( -priority, thing ))
    
    def size(self):
        return self.pq.qsize()


def my_argmax(arr, axis = -1):
    if axis == -1:
        return np.unravel_index(np.argmax(arr), arr.shape)
    elif type(axis) == int:
        return np.unravel_index(np.argmax(arr, axis = axis), arr.shape)
    elif type(axis) == tuple:
        raise Exception('not yet ready for multiple axes :/')
    else:
        raise Exception('axis data type?')

def my_argmin(arr, axis = None):
    return np.unravel_index(np.argmin(arr, axis = axis), arr.shape)


def my_choice(stuff, p = None):
    if type(stuff) == list or type(stuff) == tuple or type(stuff) == range:
        index = nra.choice(len(stuff), p = p)
        return stuff[index]
    elif type(stuff) == set or type(stuff) == hash_set:
        if p == None:
            return nra.choice(list(stuff))
        else:
            raise Exception('wtf is p?')
    elif type(stuff) == np.ndarray:
        N = np.prod(stuff.shape)
        if p == None:
            return nra.choice(stuff.reshape(N))
        elif stuff.shape == p.shape:
            return nra.choice(stuff.reshape(N), p = p.reshape(N))
        else:
            raise Exception('stuff : p mismatch')
    else:
        raise Exception('unrecognized type')


def my_csv(data, name):
    # creates csv with name filename.csv
    np.savetxt(name + '.csv', data, delimiter = ', ', fmt ='% s')


def my_dump(cucumber, name, autop = True):
    # saves file with name name.p
    file = None
    if autop:
        file = open(name + '.p', 'wb')
    else:
        file = open(name, 'wb')
    pk.dump(cucumber, file)
    file.close()


def my_entropy(somelist):
    # need not be normalized
    somelist = array(somelist)
    if np.any(somelist < 0):
        raise Exception('negative in prob array')
        
    tote = np.sum(somelist)
    probs = somelist/tote
    probs = probs + (probs == 0)
    return - np.sum(probs*log(probs))


def my_hist(stuff, xlabel = None, ylabel = None, title = None, axis = None):
    # stuff = a list of numbers
    # returns nothing, plots histogram of stuff with my fav binning
    if axis is None:
        plt.hist(stuff, bins = int(len(stuff)**0.5), histtype = 'step')
        plt.xlabel( xlabel )
        plt.ylabel(ylabel)
        plt.title(title)
    else:
        axis.hist(stuff, bins = int(len(stuff)**0.5), histtype = 'step')


def my_hstack(*mats):
    are_sparse = [sps.issparse(m) for m in mats]
    if all(are_sparse):
        return sps.hstack(mats)
    elif any(are_sparse):
        raise Exception('mixing dense and sparse?')
    else:
        box_mats = [mat.reshape(mat.shape[0], 1) if len(mat.shape) == 1 else mat for mat in mats]
        return np.concatenate(box_mats, axis = 1)


def my_kron(*mats):
    # I,B --> B 0
    #         0 B (big dimensions first)
    # I think it's better to put the little matrices first
    hold = mats[len(mats)-1]
    for a in range(len(mats)-2,-1,-1):
        hold = kron(mats[a],hold)
    return(hold)


def my_load(name, autop = True):
    file = None
    if autop:
        file = open(name + '.p', 'rb')
    else:
        file = open(name, 'rb')
    obj = pk.load(file)
    file.close()
    return obj


def my_argmax(arr, axis = -1):
    if axis == -1:
        return np.unravel_index(np.argmax(arr), arr.shape)
    elif type(axis) == int:
        return np.unravel_index(np.argmax(arr, axis = axis), arr.shape)
    elif type(axis) == tuple:
        raise Exception('not yet ready for multiple axes :/')
    else:
        raise Exception('axis data type?')


def my_max(arr, index_type = 3):
    # index_type = 1 gives raveled index
    # index_type = 2 gives tuple
    # index_type = 3 gives int for 1d guys and tuple for higher d guys
    index = np.argmax(arr)
    tuple_index = np.unravel_index(index, arr.shape)
    val = arr[tuple_index]
    if index_type == 1 or index_type == 3 and len(tuple_index) == 1:
        return val, index
    else:
        return val, tuple_index


def my_min(arr, index_type = 3):
    val, dex = my_max(-arr, index_type = index_type)
    return -val, dex


def my_outer(a, b = None):
    # returns |a><b|
    if b is None :
        b = a
    return(np.outer(a,b.conj()))


def my_savefig(fig, name):
    max_count = 10000
    save_dir = 'figs_saved'
    ext='.svg'

    # Ensure they exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists('savefig_count.p'):
        my_dump(1000, 'savefig_count')

    # Load the current count
    count = my_load('savefig_count')

    if count > max_count:
        raise Exception('Figure count exceeded max_count. If this was intentional, please increase the max_count.')

    # Increment the count and save it
    my_dump(count + 1, 'savefig_count')

    # Save the figure with the updated count
    fig.savefig(save_dir + '/' + name + '_' + str(count) + ext)


def my_shuf(arr, axis = 0):
    # returns deepcopy of array shuffled along said axis
    arr = np.swapaxes(arr, 0, axis)
    nra.shuffle(arr)
    arr = np.swapaxes(arr, 0, axis)
    return arr


def my_spar_kron(*mats):
    hold = mats[len(mats)-1]
    for a in range(len(mats)-2,-1,-1):
        hold = sps.kron(mats[a],hold)
    return(hold)


def my_vstack(*mats):
    are_sparse = [sps.issparse(m) for m in mats]
    if all(are_sparse):
        return sps.vstack(mats)
    elif any(are_sparse):
        raise Exception('mixing dense and sparse?')
    else:
        return np.concatenate(mats, axis = 0)
    
    
########################################################################################################################

# catz

def cat(*mats):
    if not all([ m.ndim == 1 for m in mats ]) :
        raise Exception('cat is only for 1d vectors')
        
    return np.concatenate(mats)


def dcat(*mats):
    my_mats = []
    for mat in mats:
        if mat.ndim == 1 :
            raise Exception('dcat doesnt do 1d vectors')
        elif mat.ndim == 2:
            my_mats.append( mat.reshape(mat.shape[0], mat.shape[1], 1) )
        else:
            my_mats.append(mat)
            
    return np.concatenate(my_mats, axis = 2)


def hcat(*mats):
    my_mats = []
    for mat in mats:
        if mat.ndim == 1 :
            my_mats.append( mat.reshape(len(mat), 1) )
        else:
            my_mats.append(mat)
    
    return np.concatenate(my_mats, axis = 1)


def vcat(*mats):
    my_mats = []
    for mat in mats:
        if mat.ndim == 1 :
            my_mats.append(mat.reshape(1, len(mat)))
        else:
            my_mats.append(mat)
    
    return np.concatenate(my_mats, axis = 0)


########################################################################################################################

# 6: funky fucks

def array_of_lists(*dims):
    L = np.prod(dims)
    
    guy = [ [] for i in range(L) ]
    guy[0] = None
    guy = array(guy, dtype = object)
    guy[0] = []
    guy = guy.reshape(dims)
    
    return guy

def bentropy(p):
    if np.any(p < 0) or np.any(p > 1):
        raise Exception('p are not probabilities')
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        return nan_to_num( - p * log(p) - (1-p) * log(1-p) )

def chop(myarray, mymin = -np.inf, mymax = np.inf):
    mytype = type(myarray)
    myarray = array(myarray)
    if mymin > mymax:
        raise Exception('flipsies flopes')
    
    if mymin == -np.inf:
        bottom = zeros(myarray.shape)
    else:
        bottom = (myarray < mymin) * mymin
    if mymax == np.inf:
        top = zeros(myarray.shape)
    else:
        top = (myarray > mymax) * mymax
    middle = (mymin <= myarray) * myarray * (myarray <= mymax)
    
    myout = top + middle + bottom
    
    if mytype == np.ndarray:
        return myout
    elif mytype == list:
        return myout.tolist()
    elif mytype == tuple:
        raise Exception('tuple not yet supported')
    else:
        raise Exception('datatype?')


def color_wheel(N):
    h = linspace(0, (N-1) / N, N)
    s = ones(N)
    v = ones(N)
    hsv = my_hstack(h,s,v)
    return mpc.hsv_to_rgb(hsv)

def get_subgraphs(adj_mat):
    # con_mat = adjacency matrix, zero when two guys arent connected
    # returns [ [subgraph indices] ]
    if adj_mat.shape[0] != adj_mat.shape[1] :
        raise Exception('adjacency matrix should be square')
    
    L, L = adj_mat.shape
    adj_mat = adj_mat.astype(bool)
    adj_mat = adj_mat + adj_mat.T
    
    groups = []
    remains = set(range(L))
    
    while len(remains) > 0:
        g = []
        horizon = set()
        horizon.add(remains.pop())
        while len(horizon) > 0:
            print(remains)
            print(horizon)
            h = horizon.pop()
            g.append(h)
            for i in range(L):
                if adj_mat[h, i] :
                    remains.remove(i)
                    horizon.add(i)
                    adj_mat[i, h] = 0
                    adj_mat[h, i] = 0
        groups.append(g)
        
    return groups
                
def get_subgraphs(adj_mat):
    # con_mat = adjacency matrix, zero when two guys arent connected
    # returns [ [subgraph indices] ]
    if adj_mat.shape[0] != adj_mat.shape[1] :
        raise Exception('adjacency matrix should be square')
    
    L, L = adj_mat.shape
    adj_mat = adj_mat.astype(bool)
    adj_mat = adj_mat + adj_mat.T
    
    pointers = list(range(L))
    groups = { i : [i] for i in range(L) }
    
    for i in range(L):
        for j in range(i):
            if adj_mat[i, j] and pointers[i] != pointers[j] :
                p = pointers[i]
                groups[p] = groups[p] + groups.pop(pointers[j])
                for k in groups[p]:
                    pointers[k] = p

    return list(groups.values())

def isvec(nparr):
    # this will return True if array is 1d, even if it is 1x1
    # something strange might happen if a dimension is zero
    flag = False
    for dim in nparr.shape:
        if dim != 1:
            if flag:
                return False
            else:
                flag = True
    return True


def nck(n, k, exact = False):
    # if exact return int, otherwise return float
    if exact :
        return factorial(n, exact = True) // factorial(k, exact = True) // factorial(n - k, exact = True)
    else:
        return factorial(n, exact = False) / factorial(k, exact = False) / factorial(n - k, exact = False)


def near(a, b, THRESH = 10**-6):
    # in float 32, machine epsilon ~ 10**-7
    # in float 64, machine epsilon ~ 10**-16 
    if - THRESH < abs(a - b) / ( abs(a) + abs(b) ) < THRESH :
        return True
    else:
        return False


def nlm_2_bib(nlm):
    # get rid of any epub or pmcid
    tokenz = nlm.split('. ')
    authors = tokenz[0]
    title = tokenz[1]
    journal = tokenz[2]
    date = tokenz[3].split(';')[0].split(' ')
    vol_num, pages = tokenz[3].split(';')[1].split(':')
    year = date[0]
    #year, month, day, vol_num, pages = re.split(r'[ ;:]', tokenz[3])
    doi = tokenz[4].split(' ')[1]
    if tokenz[5][:4] == 'Epub':
        pmid = tokenz[6].split(' ')[1][:-1]
    else:
        pmid = tokenz[5].split(' ')[1][:-1]
    
    bib = (  '@article{' + authors.split()[0] + '_' + year[2:4] + '_' + authors.split()[-2] + ',\n    '
           + 'comments = {  },\n    '
           + 'title = {' + title + '},\n    '
           + 'author = {' + authors.replace(', ', '$$$').replace(' ', ', ').replace('$$$', ' and ') + '},\n    '
           + 'journal = {' + journal + '},\n    '
           + 'volume = {' + vol_num + '},\n    '
           #+ 'number = {' + number + '},\n    '
           + 'pages = {' + pages + '},\n    '
           + 'year = {' + year + '},\n    '
           + 'doi = {' + doi + '},\n    '
           + 'note = {PMID: ' + pmid + '}\n}' )
    
    print(bib) 
    

def rand_prob(d):
    pows = np.concatenate([1 - nra.power(range(d-1,1,-1)), rand(1)])
    remains = 1
    for a in range(d-1):
        pows[a] *= remains
        remains -= pows[a]
    return np.concatenate([pows, array([1-np.sum(pows)])])


def rref(A, tol=1.0e-12):
    # made by joni from stackoverflow
    A = deepcopy(A)
    m, n = A.shape
    i, j = 0, 0
    jb = []

    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
        k = np.argmax(np.abs(A[i:m, j])) + i
        p = np.abs(A[k, j])
        if p <= tol:
            # The column is negligible, zero it out
            A[i:m, j] = 0.0
            j += 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                A[[i, k], j:n] = A[[k, i], j:n]
            # Divide the pivot row i by the pivot element A[i, j]
            A[i, j:n] = A[i, j:n] / A[i, j]
            # Subtract multiples of the pivot row from all the other rows
            for k in range(m):
                if k != i:
                    A[k, j:n] -= A[k, j] * A[i, j:n]
            i += 1
            j += 1
    # Finished
    return A, jb


def sort_array_2_list(arr, *objs, mykey = None):
    # sorts array by values and returns list of values zipped to mutidimensional array index
    # if objs are inputted these can be used in lieu of the indexes
    dims = arr.shape
    d = len(dims)
    
    if len(objs) == 0:
        objs = tuple(range(i) for i in dims)
    
    if mykey == None:
        return sorted(zip(arr.reshape(-1), it.product(*objs)))
    else:
        return sorted(zip(arr.reshape(-1), it.product(*objs)), key = lambda x: mykey(x[0]))


def sortup(x, key = None):
    return tuple(sorted(x, key = key))

###################################################################################################

########################################################################################################################

def plot_hyperpath(hyperpath, ax = None):
    # hyperpath = d x T matrix representing position
    flag = True
    if ax is not None:
        flag = False
    
    if flag:
        fig, ax = plt.subplots()
    
    d, T = hyperpath.shape
    u_x = sin(arange(0, 2*pi, 2*pi/d))
    u_y = cos(arange(0, 2*pi, 2*pi/d))
    
    x = u_x.dot(hyperpath)
    y = u_y.dot(hyperpath)
    color = linspace(0, 1, T)

    ax.scatter(x, y, s = 1, c = color, marker = '_')
    
    if flag:
        plt.show()

def split_sets(test_frac, *stuff):
    # stuff are arrays, lists, or tuples
    # if array, split will treat treat the LAST dimension as the one being shuffled over
    # returns train_stuff and test_stuff lists
    # will turn tuple into list!
    
    # ensure all objects have same number of elements
    
    L = len(stuff[0])
    if not all([len(thing) == L for thing in stuff]):
        raise Exception('shape mismatch')
    
    # pick random permutation
    
    test_num = int( test_frac * L )
    shuf = array(range(L))
    nra.shuffle(shuf)
    testers = shuf[:test_num]
    trainers = shuf[test_num:]
    
    if not ( 0 < test_num < L ):
        raise Exception('test set not between 0 and 100 %')
    
    # split sets
    
    train_stuff = []
    test_stuff = []
    for thing in stuff:
        if type(thing) == np.ndarray:
            train_stuff.append(thing[trainers])
            test_stuff.append(thing[testers])
        elif type(thing) == list or type(thing) == tuple:
            train_stuff.append([thing[i] for i in trainers])
            test_stuff.append([thing[i] for i in testers])
        else:
            raise Exception('data type')
    
    return train_stuff, test_stuff

####################################################################################################################

#### gpt code

def plot0(tit = ''):
    # Create a blank figure and axis
    fig, ax = plt.subplots()
    
    # Set the background color to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('blue')
    ax.set_title(tit, color = 'w', fontsize = 60)
    
    # Display the blank plot
    plt.show()

