import pandas as pd
import numpy as np
import pyspark
from collections import OrderedDict
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from scipy import sparse
from keras.initializers import Zeros
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import initializers


class mf_nn():
    def __init__(self, max_iter=10, regparam=0.01, k=10, rank=20, final_k=10, movie_tag_embed = False, batch_size=256, epochs=32, verbose = 0):
        self.max_iter = max_iter
        self.regparam = regparam
        self.k = k
        self.rank = rank
        self.final_k = final_k
        self.movie_tag_embed = movie_tag_embed
        self.train_y_nn = None
        self.train_X_nn = None
        self.df_tag = None
        self.train_set = None
        self.test_set = None
        self.test_y_nn = None
        self.test_X_nn = None
        self.top_k_pairs = None
        self.uf_dict = None
        self.if_dict = None
        self.nn = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.tag_dict = None
        self.history = None

    def cf(self):
        #print('matrix factorization begins......')
        spark = SparkSession.builder.appName("PySpark ALS Model").getOrCreate()
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        als = ALS(maxIter=self.max_iter, regParam=self.regparam, rank=self.rank, userCol="userId", itemCol="movieId",
                  ratingCol="rating",
                  coldStartStrategy="drop")
        #print(self.train_set.shape)
        train = spark.createDataFrame(self.train_set)
        model = als.fit(train)
        # rerive latent features from cf
        self.uf_dict = self.get_embed(model)
        self.if_dict = self.get_embed(model, 'movie')
        
        rating_train = self.train_set.pivot(index='movieId', columns='userId', values='rating')
        self.top_k_pairs = self.get_topk(self.uf_dict, self.if_dict, self.k, rating_train)

    def get_embed(self, model, type_='user'):
        if type_ == 'user':
            factors = map(lambda row: row.asDict(), model.userFactors.collect())

        if type_ == 'movie':
            factors = map(lambda row: row.asDict(), model.itemFactors.collect())

        dict_ = {x['id']: x['features'] for x in factors}
        return dict_
    


    def get_topk(self, user_dict, movie_dict, k, rating_train):

        # calculate score
        user_dict = OrderedDict(sorted(user_dict.items()))
        Xi = np.array(list(user_dict.values()))
        movie_dict = OrderedDict(sorted(movie_dict.items()))
        Yi = np.array(list(movie_dict.values()))
        score = np.dot(Xi, Yi.T)

        # make score that user already watched negative
        tmp = rating_train.copy()
        tmp.sort_index(inplace=True)
        tmp.sort_index(axis=1, inplace=True)

        tmp[~tmp.isnull()] = 10
        tmp[tmp.isnull()] = 0

        cleaned_score = score.T - np.array(tmp)

        # get top k movies
        topmovies = np.argsort(-cleaned_score, axis=0)[:k, :]

        # transform into (user,movie) pair
        users = list(user_dict.keys())
        movies = list(movie_dict.keys())

        result = []
        for u in range(topmovies.shape[1]):
            for m in topmovies[:, u]:
                result.append((users[u], movies[m]))
        return result
    
    def create_train_nn(self):
        if not self.top_k_pairs:
            self.cf()
        #print('creating training set for neural network......')
        if self.movie_tag_embed:
            feature = self.train_set.apply(lambda x: self.uf_dict[x['userId']] + self.if_dict[x['movieId']] + self.tag_dict[x['movieId']], axis=1)
        else:
            feature = self.train_set.apply(lambda x: self.uf_dict[x['userId']] + self.if_dict[x['movieId']], axis=1)
        self.train_X_nn = np.array(list(feature.values))
        self.train_y_nn = self.train_set.iloc[:, -1].values
        print('training set created')
        #print(f'shape of features array in train set: {self.train_X_nn.shape}')
        #print(f'shape of target array in train set: {self.train_y_nn.shape}')

    def create_test_nn(self):
        df_pairs = pd.DataFrame(np.array(self.top_k_pairs), columns=['userId', 'movieId'])
        self.test_set = self.test_set.append(df_pairs)
        print('creating testing set for neural network.....')
        self.test_set['pair'] = self.test_set.apply(lambda x: (x['userId'], x['movieId']), axis=1)
        self.test_set = self.test_set[self.test_set['pair'].isin(self.top_k_pairs)]
        self.test_set.drop_duplicates(subset=['userId', 'movieId'], inplace=True)
        #print(self.test_set.shape)
        self.test_set.drop(columns='pair', inplace=True)
        
        if self.movie_tag_embed:
            feature = self.test_set.apply(lambda x: self.uf_dict[x['userId']] + self.if_dict[x['movieId']] + self.tag_dict[x['movieId']], axis=1)
        else:
            feature = self.test_set.apply(lambda x: self.uf_dict[x['userId']] + self.if_dict[x['movieId']], axis=1)
            
        self.test_X_nn = np.array(list(feature.values))
        self.test_y_nn = self.test_set.iloc[:, -1].values
        print('prediction finished')
        #print(f'shape of features array in test set: {self.test_X_nn.shape}')
        #print(f'shape of target array in test set: {self.test_y_nn.shape}')

    def train_nn(self):
        print('start training neural network......')
        if self.movie_tag_embed:
            self.nn = Sequential([
                Dense(256, input_shape=(self.train_X_nn.shape[1],)),
                Activation('relu'),
                Dense(64),
                Activation('relu'),
                Dense(8),
                Activation('relu'),
                Dense(1)
            ])
        else:
            self.nn = Sequential([
                Dense(20, input_shape=(self.train_X_nn.shape[1],)),
                Activation('relu'),
                Dense(8),
                Activation('relu'),
                Dense(1)])
        self.nn.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        self.history = self.nn.fit(self.train_X_nn, self.train_y_nn, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1, verbose = self.verbose)
        print('model training finished')

    def pred_nn(self):
        return self.nn.predict(self.test_X_nn)

    def fit(self, train_df, df_tag = None):
        if self.movie_tag_embed and df_tag is None:
            raise ValueError('df_tag cannot be None')
        if df_tag is not None:
            self.tag_dict = df_tag.groupby('movieId')['relevance'].apply(lambda x: list(x)).to_dict()
        self.train_set = train_df.copy().drop('timestamp', axis = 1)
        print('Training begins.......')
        self.create_train_nn()
        self.train_nn()

    def pred(self, test_df):
        self.test_set = test_df.copy().drop('timestamp', axis = 1)
        self.create_test_nn()
        self.pred_set = self.test_set.copy()
        self.pred_set['rating'] = self.pred_nn()
        self.pred_set = self.pred_set.sort_values(['userId', 'rating']).groupby('userId').head(self.final_k)
        return self.pred_set