import pandas as pd
import numpy as np
from collections import defaultdict,OrderedDict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
import math
import random
import matplotlib.pyplot as plt
from scipy import sparse 
import seaborn as sns
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 


def sample_df(df, user_thresh=20, item_thresh=500, user_sample_n=20000, item_sample_n = 1000, random_seed=0):
    np.random.seed(random_seed)
    countItem = df[['movieId', 'rating']].groupby(['movieId']).count()
    selectedItemId = countItem.loc[countItem['rating'] > item_thresh].index
    #selectedItemId = random.sample(list(selectedItemId), item_sample_n)
    selectedItemId = np.random.choice(selectedItemId, item_sample_n, replace=False)
    df_sample_item = df[df['movieId'].isin(selectedItemId)]


    countUser = df_sample_item[['userId', 'rating']].groupby(['userId']).count()
    selectedUserId = countUser.loc[countUser['rating'] > user_thresh].index
    #selectedUserId = random.sample(list(selectedUserId), user_sample_n)
    selectedUserId = np.random.choice(selectedUserId, user_sample_n, replace=False)
    df_sample = df_sample_item[df_sample_item['userId'].isin(selectedUserId)]    
    
    n_users = len(df_sample.userId.unique())
    n_items = len(df_sample.movieId.unique())
    n_ratings = len(df_sample) 
    print(f'number of users: {n_users}')
    print(f'number of items: {n_items}')
    print(f'number of ratings: {n_ratings}')
    return df_sample


def train_test_split_by_time(df, test_size=0.2):
    '''
    for each user, use the latest 20% ratings as testing set and oldest 80% as training set
    :param df: original data
    :param test_size: proportion of test set
    :return: a tuple with train set and test set
    '''
    train_df = df.groupby('userId').apply(lambda x: x.nsmallest(math.ceil(len(x) * (1-test_size)), 'timestamp')).reset_index(drop=True)
    test_df = df.groupby('userId').apply(lambda x: x.nlargest(math.floor(len(x) * test_size), 'timestamp')).reset_index(drop=True)
    return train_df, test_df



def describe(df):
    user_freq = df.groupby('userId').count()['movieId']
    #print("Number of users: ", len(user_freq))
    movie_freq = df.groupby('movieId').count()['userId']
    #print("Number of movies: ", len(movie_freq))
    return user_freq, movie_freq


def group(df, to_group, thres= [5, 10, 50, 100, 500, 1000, 10000, 100000]):
    def get_group(x, thres=thres):
        for i in range(len(thres)):
            if x <= thres[i]:
                return i+1
            
    def get_group_range(x, thres=thres):
        for i in range(len(thres)):
            if x <= thres[i]:
                if i == 0:
                    return f'0-{thres[i]}'
                else:
                    return f'{thres[i-1]}-{thres[i]}'
            elif x > thres[-1]:
                return f'{thres[-1]}+'
    if to_group == 'user':
        df = df[['userId', 'movieId']].groupby('userId').count()
        df['groupId'] = df['movieId'].apply(lambda x: get_group(x))
        df['groupRange'] = df['movieId'].apply(lambda x: get_group_range(x)) 
        df = df.drop(['movieId'], axis = 1)
        df = df.reset_index()
        return df
    if to_group == 'movie':
        df = df[['movieId', 'userId']].groupby('movieId').count()
        df['groupId'] = df['userId'].apply(lambda x: get_group(x))
        df['groupRange'] = df['userId'].apply(lambda x: get_group_range(x)) 
        df = df.drop(['userId'], axis = 1)
        df = df.reset_index()
        return df

def plot_df(user_freq, movie_freq, user_cnt, movie_cnt, user_grouped, movie_grouped):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15,8))
    ax1, ax2, ax3, ax4 = axes.ravel()
    sns.set(style="ticks")
    sns.despine(fig=fig)

    sns.distplot(user_freq, kde=False, norm_hist=False, bins=1000, ax=ax1)
    ax1.set_xlim([0, 400])
    ax1.set_xlabel(None)

    ax1.set_title('#Ratings Across Users', size=15)
    sns.distplot(movie_freq, kde=False, norm_hist=False, bins=5000, ax=ax2)
    ax2.set_xlim([0, 200])
    ax2.set_xlabel(None)
    ax2.set_title('#Ratings Across Movies', size=15)


    groupIds = sorted(user_grouped.groupId.unique())
    groupRanges = [user_grouped[user_grouped.groupId==groupId].groupRange.unique()[0] for groupId in groupIds]

    sns.barplot(x=user_cnt['groupId'], y=user_cnt['count'], ax=ax3)
    ax3.set_xticklabels(groupRanges )  #['0-5', '5-10', '10-50', '50-100', '100-500', '500-1k', '1k-10k', '10k-100k'] 
    ax3.set_xlabel(None)

    groupIds = sorted(movie_grouped.groupId.unique())
    groupRanges = [movie_grouped[movie_grouped.groupId==groupId].groupRange.unique()[0] for groupId in groupIds]

    sns.barplot(x=movie_cnt['groupId'], y=movie_cnt['count'], ax=ax4)
    ax4.set_xticklabels(groupRanges )  #['0-5', '5-10', '10-50', '50-100', '100-500', '500-1k', '1k-10k', '10k-100k'] 
    ax4.set_xlabel(None)

    plt.tight_layout()
    plt.show()


def reco2user(matrix, k_items=5):
    #recommend top k movies for each user
    n_ratings = list(pd.DataFrame(matrix).count(axis=0))
    matrix = np.array(matrix)
    reco = np.zeros(matrix.shape)
    m = matrix.shape[1]
    topK = np.argsort(-matrix, axis=0)[:k_items,:]
    for i in range(m):
        if n_ratings[i] < k_items:
            ratings_array = matrix[:,i]
            #indice = ratings_array[ratings_array > -1]
            reco[:,i][ratings_array > -1] = 1  #only rated movies are reco
        else:      
            indice = topK[:,i]        # the k movies that are recommended to user i
            reco[:,i][indice] = 1     # set the recommended movies as 1 
    return reco
    

def recall_over_user(rating_test_df, rating_pred_matrix, k_items=5, plot=True):
    '''
    calculate the recall over users
    :param rating_test: the test rating dataframe
    :param rating_pred_matrix: the predicted rating matrix
    :return: a list of user recalls, mean recall score 
    '''
    reco_actual = reco2user(rating_test_df, k_items=k_items)
    reco_pred = reco2user(rating_pred_matrix, k_items=k_items)
    evaluation = np.zeros(reco_actual.shape)
    evaluation[(reco_actual==1) & (reco_pred==1)] = 1
    # number of right recommendations
    good_reco = evaluation.sum(axis=0)  
    # number of relevant movies that users did rate in test set
    relevant_movies = np.zeros(rating_test_df.shape[1]) 
    relevant_movies[:]=k_items
    rating_cnt = np.array(rating_test_df.count(axis=0))
    relevant_movies[rating_cnt<k_items] = rating_cnt[rating_cnt<k_items]   #if less than k movies in test, take actual number
    # recall = good_reco / relevant_movies
    user_recalls = good_reco/relevant_movies
    #user_recalls[user_recalls > 1] = 1       #recall score could not exceed 1
    avg_recall = user_recalls.mean()
    if plot:
        # plot recall distribution
        plt.hist(user_recalls)
        plt.title('Recall Distribution Over Users ')
        plt.xlabel('recall')
        plt.ylabel('number of users')
        plt.show() 
    return user_recalls, avg_recall

def recall_over_user_loose(rating_test_df, rating_pred_matrix, k_pred=50, k_relevant=20, plot=True):
    '''
    calculate the recall over users
    :param rating_test: the test rating dataframe
    :param rating_pred_matrix: the predicted rating matrix
    :return: a list of user recalls, mean recall score 
    '''
    reco_actual = reco2user(rating_test_df, k_items=k_pred)
    reco_pred = reco2user(rating_pred_matrix, k_items=k_relevant)
    # number of right recommendations
    evaluation = np.zeros(reco_actual.shape)
    evaluation[(reco_actual==1) & (reco_pred==1)] = 1
    good_reco = evaluation.sum(axis=0)  
    # number of relevant movies that users did rate in test set
    relevant_movies = np.zeros(rating_test_df.shape[1]) 
    relevant_movies[:]=k_relevant
    #if less than k_relevant movies in test, take actual number
    rating_cnt = np.array(rating_test_df.count(axis=0))
    relevant_movies[rating_cnt<k_relevant] = rating_cnt[rating_cnt<k_relevant]   
    # recall = good_reco / relevant_movies
    user_recalls = good_reco/relevant_movies
    #user_recalls[user_recalls > 1] = 1       #recall score could not exceed 1
    avg_recall = user_recalls.mean()
    if plot:
        # plot recall distribution
        plt.hist(user_recalls)
        plt.title('Recall Distribution Over Users ')
        plt.xlabel('recall')
        plt.ylabel('number of users')
        plt.show() 
    return user_recalls, avg_recall


def recall_over_user_group(rating_test, rating_pred, user_grouped, k_items=5, plot=True):
    user_recalls, avg_recall = recall_over_user(rating_test, rating_pred, k_items=5, plot=False)
    groupIds = sorted(user_grouped.groupId.unique())
    groupRanges = [user_grouped[user_grouped.groupId==groupId].groupRange.unique()[0] for groupId in groupIds]
    gRecalls = list()
    for groupId in groupIds:
        idx  = user_grouped[user_grouped.groupId == groupId].index
        gRecall = np.mean(user_recalls[idx]).round(4)
        gRecalls.append(gRecall)
        
    recall_dict = {key:value for key, value in zip(groupRanges, gRecalls)}
    print("Recall for each user group: ", recall_dict)
    _, ax = plt.subplots(1,1)
    x = np.arange(len(groupIds)) 
    y = gRecalls
    
    if plot:
        sns.barplot(x, y, ax = ax )
        ax.set_xticklabels(groupRanges)
        ax.set_title('Recall Over Different User Groups', size=15)
        ax.set_xlabel('User Group With #ratings', size=12)
        plt.show()
    return recall_dict
    

def cal_coverage(test, pred, k_items=10, threshold=3, coverage_type='user', verbose = True):
    test = np.array(test)
    # get recommendation matrix based on both actual and predictions
    reco_actual = reco2user(test, k_items=k_items)
    reco_pred =  reco2user(pred, k_items=k_items)
    # count the right coverage
    coverage = (reco_pred == reco_actual) & (reco_pred == 1) & (reco_actual == 1)
    if coverage_type == 'user':
        right_reco = np.sum(coverage, axis=0)
        user_coverage = np.sum(np.where(right_reco>=threshold,1,0))/len(right_reco)
        if verbose:
            print(f'User coverage:  the fraction of users for which at least {threshold} items can be recommended well is {user_coverage:.3f}')
        return user_coverage
    elif coverage_type == 'item':
        right_reco = np.sum(coverage, axis=1)
        item_coverage = np.sum(np.where(right_reco>=threshold,1,0))/len(right_reco)
        if verbose:
            print(f'Item coverage: the fraction of items that can be recommended to at least {threshold} users well is {item_coverage:.3f}')
        return item_coverage
    elif coverage_type == 'catalog':
        right_reco = np.sum(coverage, axis=1)
        catalog_coverage = np.sum(np.where(right_reco>=1,1,0))/len(right_reco)
        if verbose:
            print(f'Catalog coverage: the fraction of items that are in the top-{k_items} for at least 1 user is {catalog_coverage:.3f}')
        return catalog_coverage


def MAE(test, pred):
    return np.nanmean(np.absolute(test-pred))


def MSE(test, pred):
    return np.nanmean((test-pred)**2)


def RMSE(test, pred):
    return np.nanmean((test-pred)**2)**0.5


def dcg(test, pred):
    n, m = test.shape[0], test.shape[1]
    sorted_indice = np.argsort(-pred, axis=0)
    sorted_test = np.zeros(test.shape)
    ranking = np.zeros(test.shape)
    for i in range(m):
        sorted_test[:, i] = test[:, i][sorted_indice[:, i]]
    for j in range(n):
        ranking[j, :] = j + 1
    return np.mean(np.nansum((np.where(np.isnan(sorted_test), np.nan, 1))/np.log2(ranking+1), axis=1))


def ndcg(test, pred):
    test = np.array(test)
    pred = np.array(pred)
    return dcg(test, pred) / dcg(test, test)



def get_embed(model,type_='user'):
    if type_ =='user':
        factors = map(lambda row:row.asDict(),model.userFactors.collect()) 
       
    if type_ =='movie':
        factors = map(lambda row:row.asDict(),model.itemFactors.collect())  
        
    dict_ = {x['id']:x['features'] for x in factors}
    return dict_  



def get_rating_pred(model, predictions, rating_train):
    pred = predictions.toPandas()
    rating_pred= pred.pivot(index='movieId', columns='userId', values='rating')
    user_dict = get_embed(model,type_='user')
    movie_dict =get_embed(model,type_='movie')
    user_dict = OrderedDict(sorted(user_dict.items())) 
    Xi = np.array(list(user_dict.values()))
    movie_dict = OrderedDict(sorted(movie_dict.items()))  
    Yi = np.array(list(movie_dict.values()))
    rating_pred = np.dot(Xi,Yi.T).T 
    rating_pred[~rating_train.isnull()]='nan'
    return rating_pred



def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10,10))
    plt.subplot(211)
    x = np.arange(len(history.history['mae']))+1
    y_train = history.history['mae']
    y_test = history.history['val_mae']
    plt.plot(x, y_train)
    plt.plot(x, y_test)
    plt.title('MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    #plt.yticks(np.arange(3,10)/10)
    plt.legend(['Train', 'Test'], loc='upper left') 


