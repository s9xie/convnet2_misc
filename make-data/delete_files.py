import os
import pickle
import sys


print "usage: python xxx.py corrupted_list(pickle) Training.tar_path"
corrupted_list = sys.argv[1]
tar_path = sys.argv[2]
clist = pickle.load(open(corrupted_list,'r'))
for name in clist:
    #print name
    #cmd = 'tar --delete -f ../../cars_raw/Training.tar '+name
    #cmd = 'tar --delete -f ' + tar_path + ' ' + name
    cmd = 'rm ' + tar_path + name
    print cmd
    os.system(cmd)

