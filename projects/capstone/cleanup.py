import os
basedir = "C:/Users/pushkar/ML/machine-learning/projects/capstone/images/train"
dirs = os.listdir(basedir)

for d in dirs:
    dfiles = os.listdir(basedir + "/" + d)
    for i in range(1900,len(dfiles)):
        os.remove(basedir + "/" + d + "/" + dfiles[i])
    dfiles = os.listdir(basedir + "/" + d)
    print(len(dfiles))
