import pandas as pd
import numpy as np
import math
import time
from tqdm import tqdm
from scipy import sparse
from collections import defaultdict,OrderedDict
np.warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 
from keras.initializers import Zeros
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import initializers
from keras.callbacks import EarlyStopping

class ann_nn():
    def __init__(self, max_iter=10, regparam=0.01, prerank_k=30, leaf_size=30, rank=20, batch_size=256, epochs=32, k_items=10, use_element_embed=False, verbose=0):
        self.use_element_embed = use_element_embed
        self.max_iter =  max_iter
        self.regparam = regparam 
        self.rank = rank
        self.leaf_size = leaf_size
        self.userIds = None
        self.movieIds = None 
        self.n_users = None 
        self.n_movies = None
        self.user_dict = None
        self.movie_dict = None
        self.movie_matrix = None
        self.reco_matrix =  None
        self.train_set = None
        self.test_set = None
        self.pred_set = None
        self.rating_reco = None
        self.rating_train = None
        self.rating_test = None
        self.top_k_pairs = None
        self.tag_dict = None
        #nn
        self.verbose = verbose
        self.df_tag = None
        self.train_y_nn = None
        self.train_X_nn = None
        self.test_y_nn = None
        self.test_X_nn = None
        self.nn = None
        self.batch_size = batch_size
        self.epochs = epochs
        #number of reco in each stage
        self.prerank_k = prerank_k
        self.k_items = k_items


    def create_train_nn(self, df):
        self.train_set = df.copy().drop('timestamp', axis = 1)
        self.rating_train = self.train_set.pivot(index='movieId', columns='userId', values='rating')
        if not self.top_k_pairs:
            self.ann()
        print('creating training set for neural network......')
        if self.use_element_embed:
            feature = self.train_set.apply(lambda x: self.user_dict[x['userId']] + self.movie_dict[x['movieId']] + self.tag_dict[x['movieId']], axis=1)
        else:
            feature = self.train_set.apply(lambda x: self.user_dict[x['userId']] + self.movie_dict[x['movieId']], axis=1)      
        self.train_X_nn = np.array(list(feature.values))
        self.train_y_nn =self.train_set.iloc[:, -1].values
        print(f'shape of features array in train set: {self.train_X_nn.shape}')
        print(f'shape of target array in train set: {self.train_y_nn.shape}')

    def create_test_nn(self, df):        
        self.test_set = df.copy().drop('timestamp', axis = 1)
        df_pairs = pd.DataFrame(np.array(self.top_k_pairs), columns=['userId', 'movieId'])
        self.test_set = self.test_set.append(df_pairs)
        print('creating testing set for neural network.....')
        self.test_set['pair'] = self.test_set.apply(lambda x: (x['userId'], x['movieId']), axis=1)
        self.test_set = self.test_set[self.test_set['pair'].isin(self.top_k_pairs)]
        self.test_set.drop_duplicates(subset=['userId', 'movieId'], inplace=True)
        print(self.test_set.shape)
        self.test_set.drop(columns='pair', inplace=True)        
        if self.use_element_embed:
            feature = self.test_set.apply(lambda x: self.user_dict[x['userId']] + self.movie_dict[x['movieId']] + self.tag_dict[x['movieId']], axis=1)
        else:
            feature = self.test_set.apply(lambda x: self.user_dict[x['userId']] + self.movie_dict[x['movieId']], axis=1)
        self.test_X_nn = np.array(list(feature.values))
        self.test_y_nn = self.test_set.loc[:,'rating']
        print(f'shape of features array in test set: {self.test_X_nn.shape}')
        print(f'shape of target array in test set: {self.test_y_nn.shape}')

    def train_nn(self, plot=False):
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
        if self.use_element_embed:       
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
        self.history = self.nn.fit(self.train_X_nn, self.train_y_nn, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1,  verbose=self.verbose)


    def pred_nn(self):
        return self.nn.predict(self.test_X_nn)


    def fit(self, train_df, df_tag=None):
        print('Training begins.......')
        if df_tag is not None:
            self.tag_dict = df_tag.groupby('movieId')['relevance'].apply(lambda x: list(x)).to_dict()
        self.create_train_nn(train_df)
        self.train_nn()

        
    def pred(self, test_df):
        print('Prediction begins.......')
        self.create_test_nn(test_df)
        self.pred_set = self.test_set.copy()
        self.pred_set['rating'] = self.pred_nn()
        self.pred_set.sort_values(['userId', 'rating']).groupby('userId').head(self.k_items)
        return self.pred_set


    def get_embed(self, model,type_='user'):
        if type_ =='user':
            factors = map(lambda row:row.asDict(),model.userFactors.collect()) 
           
        if type_ =='movie':
            factors = map(lambda row:row.asDict(),model.itemFactors.collect())  
            
        dict_ = {x['id']:x['features'] for x in factors}
        return dict_  


    def cf(self):
        print('Begin to retrieve latent features from MF ......')
        spark = SparkSession.builder.appName("PySpark ALS Model").getOrCreate()
        #spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        als = ALS(maxIter=self.max_iter, regParam=self.regparam, rank=self.rank, userCol="userId", itemCol="movieId",
                  ratingCol="rating",
                  coldStartStrategy="drop")
        #print(self.train_set.shape)
        train = spark.createDataFrame(self.train_set)
        model = als.fit(train)
        # retrieve latent features from cf
        user_dict = self.get_embed(model,type_='user')
        user_dict = OrderedDict(sorted(user_dict.items())) 
        self.user_dict = user_dict
        movie_dict = self.get_embed(model,type_='movie')
        movie_dict = OrderedDict(sorted(movie_dict.items()))  
        self.movie_dict = movie_dict
        self.movie_matrix = np.array(list(self.movie_dict.values()))


    def ann(self):
        self.rating_train = self.train_set.pivot(index='movieId', columns='userId', values='rating')
        self.userIds = np.sort(self.train_set.userId.unique())
        self.movieIds = np.sort(self.train_set.movieId.unique())
        self.n_users = len(self.userIds)
        self.n_movies = len(self.movieIds)
        self.reco_matrix = np.zeros((self.n_movies, self.n_users))
        if not self.movie_matrix:
            self.cf()
        print(f"Begin to retrieve top {self.prerank_k} pairs with ann......")
        self.top_k_pairs = []
        for i in range(self.n_users):
            userId = self.userIds[i]  
            userEmbed = np.array(self.user_dict[userId])
            userEmbed = np.array([userEmbed])         #turns into shape (1, 10)
            compare = np.concatenate([self.movie_matrix, userEmbed])
            tree = KDTree(compare, leaf_size=self.leaf_size, metric='euclidean')              
            dist, ind = tree.query(userEmbed, k=self.prerank_k*4)  
            ind_reco = ind[0][1:]       #len=k_items, exclude first index (which is itself)
            #exclude movie ind that has ratings in rating_train
            ind_reco_array = np.array(ind_reco)
            mask = self.rating_train.iloc[ind_reco, i].notnull()
            ind_to_remove = ind_reco_array[mask]
            ind_reco_final = [ind for ind in ind_reco if ind not in ind_to_remove][:self.prerank_k]
            for j in ind_reco_final:
                movieId = self.movieIds[j]
                self.top_k_pairs.append((userId, movieId))
        print(f"top {self.prerank_k} pairs are ready")



