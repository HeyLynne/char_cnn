#coding=utf-8
import csv
import numpy as np
import traceback

from config import Config

class DataSet(object):
    def __init__(self, data_source):
        self.data_source = data_source
        self.index_in_epoch = 0
        self.alphabet = Config.alphabet
        self.alphabet_size = Config.alphabet_size
        self.num_classes = Config.nums_classes
        self.l0 = Config.l0
        self.epochs_completed = 0
        self.batch_size = Config.batch_size
        self.example_nums = Config.example_nums
        self.doc_image = []
        self.label_image = []

    def next_batch(self):
        if self.index_in_epoch >= self.example_nums:# begin a new epoch
            self.epochs_completed += 1
            perm = np.arange(self.example_nums)
            np.random.shuffle(prem)
            self.doc_image = self.doc_image[perm]
            self.label_image = self.label_image[perm]

            self.index_in_epoch = 0
            assert self.batch_size <= self.num_classes
        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        end = min(self.index_in_epoch, self.num_classes)
        batch_x = np.array(self.doc_image[start:end], dtype='int64')
        batch_y = np.array(self.label_image[start:end], dtype='float32')

        return batch_x, batch_y

    def onehot_dic_build(self):
        alphabet = self.alphabet
        embedding_dic = {}
        embedding_w = []
        embedding_dic[Config.unkown_sign] = 0 # corner word using zero to replace
        embedding_w.append(np.zeros(len(alphabet), dtype='float32'))

        for i, alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet), dtype='float32')
            embedding_dic[alpha] = i + 1
            onehot[i] = 1
            embedding_w.append(onehot)

        embedding_w = np.array(embedding_w, dtype='float32')
        return embedding_w, embedding_dic

    def doc_process(self, doc, embedding_dic):
        min_len = min(self.l0, len(doc))
        doc_vec = np.zeros(self.l0, dtype = 'int64')
        for j in xrange(min_len):
            if doc[j] in embedding_dic:
                doc_vec[j] = embedding_dic[doc[j]]
            else:
                doc_vec[j] = embedding_dic[Config.unkown_sign]
        return doc_vec

    def dataset_process(self):
        print ("Building char dict...")
        embedding_w, embedding_dic = self.onehot_dic_build()

        print ("Processing data source...")
        doc_image = []
        label_image = []
        doc_count = 0
        csv_file = open(self.data_source, 'r')
        for line in csv.reader(csv_file, delimiter=',', quotechar='"'):
            content = line[1] + ". " + line[2]
            content = content.lower()
            doc_vec = self.doc_process(content, embedding_dic)
            doc_image.append(doc_vec)

            try:
                label = int(line[0])
            except:
                traceback.print_exc()
                continue
            label_class = np.zeros(self.num_classes, dtype = 'float32')
            label_class[label - 1] = 1
            label_image.append(label_class)
            doc_count += 1
        self.doc_image = np.asarray(doc_image, dtype='int64')
        self.label_image = np.array(label_image, dtype='float32')