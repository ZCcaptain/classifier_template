import os
import json
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional


class Weapon_resnet_preprocessing(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root

        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)


    def split_data(self):
        train_file = os.path.join(self.data_root, 'train.txt')
        dev_file = os.path.join(self.data_root, 'dev.txt')
        label_list = os.listdir(self.raw_data_root)
        with open(train_file, 'w') as t, open(dev_file, 'w') as d:
            for label in label_list:
                label_dir = os.path.join(self.raw_data_root, label)
                dir_list = os.listdir(label_dir)
                for dir in dir_list:
                    pic_dir = os.path.join(label_dir, dir)
                    pic_list = os.listdir(pic_dir)
                    count = 0
                    for pic in pic_list:
                        filename = os.path.join(pic_dir, pic)
                        result={'filename':filename, 'label':int(label)}
                        result_json = json.dumps(result)
                        if count % 10 < 8:
                            t.write(result_json)
                            t.write('\n')
                        else:
                            d.write(result_json)
                            d.write('\n')
                        count += 1


