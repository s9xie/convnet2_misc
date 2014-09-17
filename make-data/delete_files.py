import os
import pickle
import sys

#CAREFUL TO USE!!!!!!!!!OTHERWISE RM COULD KILL YOU
print "usage: python xxx.py corrupted_list(pickle) path"
corrupted_list = sys.argv[1]
path = sys.argv[2]
clist = pickle.load(open(corrupted_list,'r'))
i = 1
for name in clist:
    print i,'/',len(clist)
    print name
    #cmd = 'tar --delete -f ' + tar_path + ' ' + name  --this line is too slow. just delete the files and compress again
    cmd = 'rm ' + path + name
    #print cmd
    i = i + 1
    os.system(cmd)

