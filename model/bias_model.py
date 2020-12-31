import pandas as pd
import numpy as np
import seaborn as sns
import math
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from surprise.model_selection import GridSearchCV
import matplotlib.pyplot as plt


class bias_model():
    def __init__(self):
        self.user_mean = None
        self.movie_mean = None
        self.train_rating_matrix = None
        self.test_rating_matrix = None
        self.rating_pred = None
        self.reco_matrix = None

    def fit(self, train_df):
        self.train_rating_matrix = train_df.pivot(index='movieId', columns='userId', values='rating')
        #print("train rating matrix size: ", self.train_rating_matrix.shape)
        self.user_mean = np.array(np.mean( self.train_rating_matrix , axis=0))  #user average rating
        self.movie_mean = np.array(np.mean( self.train_rating_matrix , axis=1)) # movie average rati
        
    def pred(self, test_df):
        self.test_rating_matrix = test_df.pivot(index='movieId', columns='userId', values='rating')
        #print(self.test_rating_matrix.shape)
        rating_pred = np.add(self.movie_mean[:, np.newaxis], self.user_mean[np.newaxis,:] )
        rating_pred[self.train_rating_matrix.notnull()] = np.NaN
        self.rating_pred = rating_pred
        return rating_pred
    
    def reco(self, test_df, k_items):
        rating_pred = self.pred(test_df)
        reco_matrix = reco2user(rating_pred, k_items=k_items)  #user-movie reco matrix (0 and 1)
        self.reco_matrix = reco_matrix
        return reco_matrix


