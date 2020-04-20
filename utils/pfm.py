import numpy as np
import re
import sys

'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def load_pfm(filename):
    color = None
    width = None
    height = None
    scale = None
    endian = None

    with open(filename, 'rb') as file:
        header = file.readline().decode('utf-8').rstrip()

        if header == 'PF':
            color = True    
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().decode('utf-8').rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')

    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape)
