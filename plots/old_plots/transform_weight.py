"""
Usage:
  transform_weight.py <file_name>
"""
from docopt import docopt
import sys
import numpy as np

arguments = docopt(__doc__, version='Naval Fate 2.0')
# print(arguments)
file_name = arguments['<file_name>']

lines = np.asarray([i for i in range(1, 11)]).reshape(-1 ,1)
with open(file_name) as f:
    for line in f:
        list_elem = line.split('[')[1].split(']')[0].replace('  ', ' ').split(' ')
        list_elem_float = [float(i) for i in list_elem]
        new_col = np.asarray(list_elem_float).reshape(-1, 1)
        lines = np.concatenate((lines, new_col), 1)

with open(file_name+"_tmp", "w") as f:
    for ll in lines:
        f.write(np.array2string(ll).replace('\n', '').replace('[','').replace(']','')+"\n")

f.close