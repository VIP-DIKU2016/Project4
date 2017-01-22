#!/usr/bin/python2
# -*- coding: utf-8 -*-

import ntpath, sys, itertools
import numpy as np
import ps_utils as pu
from numpy.linalg import inv, pinv
from numpy import linalg as LA

def main(argv):

    # check if the name of program is passed as argument
    if len(argv) != 2:
        print '\nRUN: python ps.py <dataset>'
        return 1

    dataset = argv[1]

    # Call the program
    try:
        # check if dataset exists
        open(dataset, 'r')
        # check if there is a program to process this data file
        prog_str = ntpath.basename(dataset).split('.')[0]
        prog = getattr(sys.modules[__name__], prog_str)
        # run the program to process the dataset
        prog(dataset)
    except IOError:
        print "There's no file \""+dataset+"\""
    except AttributeError:
        print "There is no program to process \""+dataset+"\""
    except Exception as e:
        print 'ERROR: something went wrong \n ', prog, e

    # Return 0 to indicate normal termination
    return 0

def Beethoven(file):

    # read the imaeges, mask, and light vectors from the data file
    I, mask, S = pu.read_data_file(file)

    # reshape images vector to seperate different views
    images = I.transpose(2,0,1)

    # create the masked images vector J
    n, w, h = images.shape 
    nz = np.count_nonzero(mask)
    J = np.empty([n, nz])
    mask_list = pu.tolist(mask)
    for i, image in enumerate(images):
        J[i] = mask_img(image, mask_list)

    # calculate albedo modulated normal field
    S_inv = inv(S)
    prod = np.dot(S_inv, J)
    M = unmask_batch(prod, mask_list, [3, len(mask_list)])

    # extract albedo within the mask and transform it to 2D
    albedo = (LA.norm(M, axis=0)).reshape([w, h])

    # extract n1, n2 and n3, the components of normal field 
    N = prod/LA.norm(prod, axis=0)
    N = unmask_batch(N, mask_list, [3, len(mask_list)])
    n1, n2, n3 = N.reshape([3, w, h])

    # calculate the depth
    depth = pu.unbiased_integrate(n1, n2, n3, mask)

    # display different views of the image 
    pu.display_images(images, albedo = False)
    # display albedo 
    pu.display_images([albedo], albedo = True)
    # display the depth
    pu.display_depth_mayavi(depth)


def Buddha(file):

    # read the imaeges, mask, and light vectors from the data file
    I, mask, S = pu.read_data_file(file)

    # reshape images vector to seperate different views
    images = I.transpose(2,0,1)

    # create the masked images vector J
    n, w, h = images.shape 
    nz = np.count_nonzero(mask)
    J = np.empty([n, nz])
    mask_list = pu.tolist(mask)
    for i, image in enumerate(images):
        J[i] = mask_img(image, mask_list)

    # calculate albedo modulated normal field
    S_inv = pinv(S)
    prod = np.dot(S_inv, J)
    M = unmask_batch(prod, mask_list, [3, len(mask_list)])

    # extract albedo within the mask and transform it to 2D
    albedo = (LA.norm(M, axis=0)).reshape([w, h])

    # extract n1, n2 and n3, the components of normal field 
    N = prod/LA.norm(prod, axis=0)
    N = unmask_batch(N, mask_list, [3, len(mask_list)])
    n1, n2, n3 = N.reshape([3, w, h])

    # calculate the depth
    depth = pu.unbiased_integrate(n1, n2, n3, mask)
    
    # display different views of the image
    pu.display_images(images, albedo = False)
    # display albedo 
    pu.display_images([albedo], albedo = True)
    # display the depth
    pu.display_depth_mayavi(depth)

# function to mask an image
def mask_img(image, mask):
    img_list = pu.tolist(image)
    return list(itertools.compress(img_list, mask))

# function to unmask an image
def unmask_img(masked, mask):
    j = 0;
    processed = np.empty(len(mask))
    for i, pixel in enumerate(mask):
        if pixel == 1:
            processed[i] = masked[j]
            j+=1

    return np.asarray(processed)

# function to unmask a number of images
def unmask_batch(imgs, mask, dim):
    processed = np.empty(dim)
    for i, img in enumerate(imgs): 
        processed[i] = unmask_img(img, mask)

    return processed


if __name__ == '__main__':    
    sys.exit(main(sys.argv))
else:
    print __name__