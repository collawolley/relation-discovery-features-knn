__author__ = 'hadyelsahar'


import re
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering
from sklearn.feature_extraction.text import CountVectorizer
from evaluation.evaluation import ClusterEvaluation
from sklearn.decomposition import PCA
import logging
from IPython.core.debugger import Tracer; debug_here = Tracer()
import scipy.sparse as sp

class Run:

    def __init__(self, input, loadfile=None):
        """

        :param input:
        :return:
        """

        if loadfile == None:
            self.__SPECIALTOKEN = "_SPECIAL_"

            # Initializing Vectorizer

            print "loading sentences"
            self.data = pd.read_csv(input, sep='\t', names=['dep', 'sub', 'obj', 'type', 'trigger', 'file', 'sentence', 'pos', 'relation'])

            print "extraction of entities..."
            Entities = np.unique(np.concatenate([self.data['sub'].unique(), self.data['obj'].unique()]))
            self.Entitiesids = np.arange(0, len(Entities), dtype=int)
            entities_dict = dict(zip(Entities, self.Entitiesids))
            self.data['es'] = self.data['sub'].map(lambda i: entities_dict[i])  # ID of the entity mention according to the dict
            self.data['eo'] = self.data['obj'].map(lambda i: entities_dict[i])  # ID of the entity mention according to the dict

            print "preprocessing sentences.."
            self.data = self.preprocess_dataset_sentences(self.data)
            self.data = self.data.reset_index()
        else:
            self.data = pd.read_pickle(loadfile)

        self.es_idset = self.data['es'].unique()
        self.eo_idset = self.data['eo'].unique()

    def run(self, STAGES="11"):
        """

        :param STAGES:  11 RUN ALL STAGES
                        10 RUN vectorization only
        :return:
        """

        if STAGES[0] == '1':
            print "Vectorization.."
            X = self.vectorize()

        if STAGES[1] == '1':
            print "Evaluation.."
            y = self.data['relation'].values
            self.evaluate(X, y)


    def preprocess_dataset_sentences(self, data):
        """
        method to preprocess the raw input sentences of the dataset.
        removing mistakenly parsed sentences  ( len > 200 )
        adapt colums in the dataframe to add adapted_sentences ( remove mentions of sub and obj entities with SUBJECT and OBJECT )

        replace entities in the sentence with their datatypes in the form of "Special_token + NERTYPE"

        :param data: input data frame as read from the csv file input
        :return: dataframe after preprocessing
        """

        data = data[data['sentence'].apply(lambda x: len(str(x).split()) < 200)]
        data = data[data['type'].apply(lambda x: len(str(x).split('-')) == 2)]

        data['sub-type'] = np.array([str(l).split('-')[0] for l in data['type'].values])
        data['obj-type'] = np.array([str(l).split('-')[1] for l in data['type'].values])

        # replacing entities with SPECIALTOKEN + entity Type
        data['sentence_tagged'] = data.apply(lambda l: re.sub(r"/\S+", r"", str(l['sentence']))
                                             .replace('/', ""), axis=1)

        data['sentence_tagged'] = data.apply(lambda l: str(l['sentence_tagged'])
                                             .replace(l['sub'], " %s%s " % (self.__SPECIALTOKEN, l['sub-type']), 1)
                                             .replace(l["obj"], " %s%s " % (self.__SPECIALTOKEN, l['obj-type']), 1)
                                             , axis=1)

        sub_pos=[]
        obj_pos=[]
        ids = []

        for i, row in enumerate(data.iterrows()):

            t = str(row[1]['sentence_tagged']).split()

            if self.__SPECIALTOKEN+row[1]['sub-type'] in t  and self.__SPECIALTOKEN+row[1]['obj-type'] in t:

                sub_pos.append(t.index(self.__SPECIALTOKEN+row[1]['sub-type']))
                obj_pos.append(t.index(self.__SPECIALTOKEN+row[1]['obj-type']))
                ids.append(i)

        data = data.iloc[ids]
        data['sub-pos'] = sub_pos
        data['obj-pos'] = obj_pos

        data = data.reset_index()
        return data

    def vectorize(self):
        """
        vectorize part of the input data
        :param start: start id
        :param end: end id
        :return: X, es, eo, es_ns, eo_ns
        """

        vectorizer = CountVectorizer(min_df=10)
        f_sub = [[int(x) for x in np.binary_repr(i, 25)] for i in self.data['es'].values]
        f_obj = sp.csr_matrix([[int(x) for x in np.binary_repr(i, 25)] for i in self.data['eo'].values])

        f_types = vectorizer.fit_transform([i for i in self.data['type'].values])
        f_subtype = vectorizer.fit_transform([i for i in self.data['sub-type'].values])
        f_objtype = vectorizer.fit_transform([i for i in self.data['sub-type'].values])
        vectorizer = CountVectorizer(min_df=50)
        f_trigger = vectorizer.fit_transform([i.decode('utf-8', 'ignore') for i in self.data['trigger'].values])
        f_pos = vectorizer.fit_transform([i for i in self.data['pos'].values])
        vectorizer = CountVectorizer(min_df=0)
        f_dist = vectorizer.fit_transform([unicode(i) for i in self.data['sub-pos'].values - self.data['obj-pos'].values])


        X = sp.hstack((
#		   f_sub,
#                   f_obj,
                   f_types,
                   f_subtype,
                   f_objtype,
#                   f_trigger,
#                   f_pos,
#                   f_dist
	           ), 
                   format='csr')

        print "returning dense"
        return X.todense()

    def evaluate(self, x, y):

        xf = x[y > 0]
        yf = y[y > 0]

        print "Starting Kmeans clustering.."
        clustering = KMeans(n_clusters=100)
        predictions_minibatch = clustering.fit_predict(xf)
        print "done Kmeans clustering.."
        print "homogeneity score = %s" % metrics.homogeneity_score(yf, predictions_minibatch)
        e = ClusterEvaluation(yf, predictions_minibatch)
        m = e.printEvaluation()
