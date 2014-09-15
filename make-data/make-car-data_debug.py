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

# Set this to True to crop images to square. In this case each image will be
# resized such that its shortest edge is OUTPUT_IMAGE_SIZE pixels, and then the
# center OUTPUT_IMAGE_SIZE x OUTPUT_IMAGE_SIZE patch will be extracted.
#
# Set this to False to preserve image borders. In this case each image will be
# resized such that its shortest edge is OUTPUT_IMAGE_SIZE pixels. This was
# demonstrated to be superior by Andrew Howard in his very nice paper:
# http://arxiv.org/abs/1312.5402
CROP_TO_SQUARE          = True
OUTPUT_IMAGE_SIZE       = 256

# Number of threads to use for JPEG decompression and image resizing.
NUM_WORKER_THREADS      = 1

# Don't worry about these.
OUTPUT_BATCH_SIZE = 1
OUTPUT_SUB_BATCH_SIZE = OUTPUT_BATCH_SIZE

def pickle(filename, data):
    with open(filename, "w") as fo:
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)

def unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents

def partition_list(l, partition_size):
    divup = lambda a,b: (a + b - 1) / b
    return [l[i*partition_size:(i+1)*partition_size] for i in xrange(divup(len(l),partition_size))]

def open_tar(path, name):
    if not os.path.exists(path):
        print "ILSVRC 2012 %s not found at %s. Make sure to set ILSVRC_SRC_DIR correctly at the top of this file (%s)." % (name, path, sys.argv[0])
        sys.exit(1)
    return tarfile.open(path)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_devkit_meta(ILSVRC_DEVKIT_TAR):
    tf = open_tar(ILSVRC_DEVKIT_TAR, 'devkit tar')
    fmeta = tf.extractfile(tf.getmember('ILSVRC2012_devkit_t12/data/meta.mat'))
    meta_mat = scipy.io.loadmat(StringIO(fmeta.read()))
    labels_dic = dict((m[0][1][0], m[0][0][0][0]-1) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
    label_names_dic = dict((m[0][1][0], m[0][2][0]) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
    label_names = [tup[1] for tup in sorted([(v,label_names_dic[k]) for k,v in labels_dic.items()], key=lambda x:x[0])]

    fval_ground_truth = tf.extractfile(tf.getmember('ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
    validation_ground_truth = [[int(line.strip()) - 1] for line in fval_ground_truth.readlines()]
    tf.close()
    return labels_dic, label_names, validation_ground_truth

def write_batches(target_dir, name, start_batch_num, labels, jpeg_files):
    jpeg_files = partition_list(jpeg_files, OUTPUT_BATCH_SIZE)
    labels = partition_list(labels, OUTPUT_BATCH_SIZE)
    makedir(target_dir)
    print "Writing %s batches..." % name
    corrupt_list = []
    for i,(labels_batch, jpeg_file_batch) in enumerate(zip(labels, jpeg_files)):
	#if i < 7851:
	#    continue
        t = time()
        jpeg_strings = list(itertools.chain.from_iterable(resizeJPEG([jpeg.read() for jpeg in jpeg_file_batch], OUTPUT_IMAGE_SIZE, NUM_WORKER_THREADS, CROP_TO_SQUARE)))
        if len(jpeg_strings) != 1:
	    print i, jpeg_file_batch[0].name, len(jpeg_strings)
	    corrupt_list.append(jpeg_file_batch[0].name)
	#batch_path = os.path.join(target_dir, 'data_batch_%d' % (start_batch_num + i))
        #makedir(batch_path)
        #for j in xrange(0, len(labels_batch), OUTPUT_SUB_BATCH_SIZE):
        #    pickle(os.path.join(batch_path, 'data_batch_%d.%d' % (start_batch_num + i, j/OUTPUT_SUB_BATCH_SIZE)), 
        #           {'data': jpeg_strings[j:j+OUTPUT_SUB_BATCH_SIZE],
        #            'labels': labels_batch[j:j+OUTPUT_SUB_BATCH_SIZE]})
        ##print "Wrote %s (%s batch %d of %d) (%.2f sec)" % (batch_path, name, i+1, len(jpeg_files), time() - t)
    pickle('car_raw_corrupted_list_testing', corrupt_list)
    print corrupt_list
    print len(corrupt_list)
    return i + 1

def generate_labels(train_tar, test_tar):

    with open_tar(test_tar, 'data tar') as tf:
        member = tf.getmembers()
        #print member[1].name
        members = [member[m].name.split('/')[1] for m in range(len(member)) if (member[m].isdir() == True and "__" in member[m].name) ]

    labels_dic = dict()
    count = 0;
    for m in members:
        labels_dic[m] = count
        count = count + 1
    #hardcoded for now

    label_names = members
    #print len(label_names)
    #print label_names
    #labels_dic['back'] = 0
    #labels_dic['backside'] = 1
    #labels_dic['front'] = 2
    #labels_dic['frontside'] = 3
    #labels_dic['side'] = 4
    #label_names = ['back', 'backside', 'front', 'frontside', 'side']

    return labels_dic, label_names


if __name__ == "__main__":
    parser = argp.ArgumentParser()
    parser.add_argument('--src-dir', help='Directory containing tar files', required=True)
    parser.add_argument('--tgt-dir', help='Directory to output data batches suitable for cuda-convnet to train on.', required=True)
    args = parser.parse_args()
    
    print "CROP_TO_SQUARE: %s" % CROP_TO_SQUARE
    print "OUTPUT_IMAGE_SIZE: %s" % OUTPUT_IMAGE_SIZE
    print "NUM_WORKER_THREADS: %s" % NUM_WORKER_THREADS

    TRAIN_TAR = os.path.join(args.src_dir, 'Training.tar')
    VALIDATION_TAR = os.path.join(args.src_dir, 'Testing.tar')

    assert OUTPUT_BATCH_SIZE % OUTPUT_SUB_BATCH_SIZE == 0
    labels_dic, label_names = generate_labels(TRAIN_TAR, VALIDATION_TAR)
    #labels_dic, label_names, validation_labels = parse_devkit_meta(ILSVRC_DEVKIT_TAR)

#    with open_tar(TRAIN_TAR, 'training tar') as tf:
#        member = tf.getmembers()
#	members = [member[m] for m in range(len(member)) if member[m].isdir() == False]
#	
#        train_jpeg_files = []
#        for i in range(len(members)):
#            if i % 100 == 0:
#                print "training data building: %d%% ..." % int(round(100.0 * float(i) / len(members))),
#                sys.stdout.flush()
#            train_jpeg_files += [tf.extractfile(members[i])]
#            
#        #shuffle(train_jpeg_files)
#        train_labels = [[labels_dic[jpeg.name.split('/')[1]]] for jpeg in train_jpeg_files]
#	print "done"
#    
#        # Write training batches
#        i = write_batches(args.tgt_dir, 'training', 0, train_labels, train_jpeg_files)
#    
#	# Write validation batches
    val_batch_start = 1000#int(math.ceil((i / 1000.0))) * 1000 #starting from 1000
    with open_tar(VALIDATION_TAR, 'validation tar') as tf:
        member = tf.getmembers()
        members = [member[m] for m in range(len(member)) if member[m].isdir() == False]
        
        validation_jpeg_files = []
        for i in range(len(members)):
            if i % 100 == 0:
                print "testing data building: %d%% ..." % int(round(100.0 * float(i) / len(members))),
                sys.stdout.flush()
            validation_jpeg_files += [tf.extractfile(members[i])]

        validation_labels = [[labels_dic[jpeg.name.split('/')[1]]] for jpeg in validation_jpeg_files]
        write_batches(args.tgt_dir, 'validation', val_batch_start, validation_labels, validation_jpeg_files)
    
    # Write meta file
    meta = unpickle('input_meta')
    meta_file = os.path.join(args.tgt_dir, 'batches.meta')
    meta.update({'batch_size': OUTPUT_BATCH_SIZE,
                 'num_vis': OUTPUT_IMAGE_SIZE**2 * 3,
                 'label_names': label_names})
    pickle(meta_file, meta)
    print "Wrote %s" % meta_file
    print "All done! Image batches are in %s" % args.tgt_dir
