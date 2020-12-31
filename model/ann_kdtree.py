import pandas as pd
import numpy as np
import math
import time
from collections import defaultdict,OrderedDict
from scipy import sparse
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



class ann_kdtree():
    def __init__(self, max_iter=20, regparam=0.1, rank=10, leaf_size=20, k_items=10):
        self.max_iter =  max_iter
        self.regparam = regparam 
        self.rank = rank
        self.leaf_size = leaf_size
        self.k_items = k_items
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
        self.rating_train = None
        self.rating_test = None

    def get_embed(self, model,type_='user'):
        if type_ =='user':
            factors = map(lambda row:row.asDict(),model.userFactors.collect()) 
           
        if type_ =='movie':
            factors = map(lambda row:row.asDict(),model.itemFactors.collect())  
            
        dict_ = {x['id']:x['features'] for x in factors}
        return dict_  


    def cf(self):
        print('Begin to retrieve latent features from cf ......')
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
        self.movie_matrix = np.array(list(movie_dict.values()))


    def fit(self, train_df):
        print("training begins ......")
        self.train_set = train_df
        self.rating_train = train_df.pivot(index='movieId', columns='userId', values='rating')
        self.userIds = np.sort(train_df.userId.unique())
        self.movieIds = np.sort(train_df.movieId.unique())
        self.n_users = len(self.userIds)
        self.n_movies = len(self.movieIds)
        self.reco_matrix = np.zeros((self.n_movies, self.n_users))
        if not self.movie_matrix:
            self.cf()
        print("training finishes.")

        
    def pred(self, test_df):
        print('prediction begins ......')
        self.test_set = test_df
        #self.rating_test = test_df.pivot(index='movieId', columns='userId', values='rating')
        for i in range(self.n_users):
            userId = self.userIds[i]  
            userEmbed = np.array(self.user_dict[userId])
            userEmbed = np.array([userEmbed])         #turns into shape (1, 10)
            compare = np.concatenate([self.movie_matrix, userEmbed])
            #print(compare.shape)
            #print(userEmbed == compare[-1])
            tree = KDTree(compare, leaf_size=self.leaf_size, metric='euclidean')              
            dist, ind = tree.query(userEmbed, k=self.k_items+1+5)   
            #print(ind)
            ind_reco = ind[0][1:]  #len=k_items, exclude first index (which is itself)
            #exclude movie ind that has ratings in rating_train
            ind_reco_array = np.array(ind_reco)
            mask = self.rating_train.iloc[ind_reco, i].notnull()
            ind_to_remove = ind_reco_array[mask]
            ind_reco_final = [ind for ind in ind_reco if ind not in ind_to_remove][:self.k_items]
            #print(ind_reco)
            #print(ind_reco.shape)
            self.reco_matrix[ind_reco_final, i] = 1
            #print(reco_matrix.sum(axis=0))
        print('prediciton finishes.')
        return self.reco_matrix
