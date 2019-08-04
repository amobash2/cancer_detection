import os
import argparse
import random
import shutil
from shutil import copyfile


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def main(config):

    rm_mkdir(config.train_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.test_path)

    filenames = os.listdir(config.origin_train_path)

    num_train = int((config.train_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*len(filenames))
    num_valid = int((config.valid_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*len(filenames))
    num_test = len(filenames) - num_train - num_valid

    print('\nNum of train set : ',num_train)
    print('\nNum of valid set : ',num_valid)
    print('\nNum of test set : ',num_test)

    Arange = list(range(len(filenames)))
    random.shuffle(Arange)

    for i in range(num_train):
        idx = Arange.pop()

        if (i+1) % 1000 == 0:
            print("Processed {} out of {} train files!".format(i+1, num_train))

        src = os.path.join(config.origin_train_path, filenames[idx])
        dst = os.path.join(config.train_path, filenames[idx])
        copyfile(src, dst)
    print("Finished processing {} train files!".format(num_train))
        

    for i in range(num_valid):
        idx = Arange.pop()

        if (i+1) % 1000 == 0:
            print("Processed {} out of {} validation files!".format(i+1, num_valid))

        src = os.path.join(config.origin_train_path, filenames[idx])
        dst = os.path.join(config.valid_path, filenames[idx])
        copyfile(src, dst)
    print("Finished processing {} validation files!".format(num_valid))

    for i in range(num_test):
        idx = Arange.pop()

        if (i+1) % 1000 == 0:
            print("Processed {} out of {} test files!".format(i+1, num_test))

        src = os.path.join(config.origin_train_path, filenames[idx])
        dst = os.path.join(config.test_path, filenames[idx])
        copyfile(src, dst)
    print("Finished processing {} test files!".format(num_test))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # data path
    parser.add_argument('--origin_train_path', type=str, default='/datadrive/Metastatic/train/')
    
    parser.add_argument('--train_path', type=str, default='/datadrive/Metastatic/dataset/train/')
    parser.add_argument('--valid_path', type=str, default='/datadrive/Metastatic/dataset/valid/')
    parser.add_argument('--test_path', type=str, default='/datadrive/Metastatic/dataset/test/')

    config = parser.parse_args()
    print(config)
    main(config)