import tarfile
from StringIO import StringIO
from random import shuffle
import sys
from time import time
from pyext._MakeDataPyExt import resizeJPEG
import itertools
import os
import cPickle
import scipy.io
import math
import argparse as argp

def open_tar(path, name):
    if not os.path.exists(path):
        print "ILSVRC 2012 %s not found at %s. Make sure to set ILSVRC_SRC_DIR correctly at the top of this file (%s)." % (name, path, sys.argv[0])
        sys.exit(1)
    return tarfile.open(path)

tf = open(ILSVRC_DEVKIT_TAR, 'devkit tar')
