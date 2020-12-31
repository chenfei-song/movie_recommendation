# 4571 Personalization Final Project

**Team member**

- Chenfei Song cs3913
- Leyi Mai  lm3504 
- Ming Jin mj2940
- Xiaoyu Chen xc2525

## Project Description

Our overall objective is to build a production-quality movie recommendation pipeline, so as to provide personalizied TOP-10 list of movies specific to our users.
These recommended movies will be displayed in the section of TOP 10 MOVIES THAT YOU MAY LIKE in the carousel on the APP or website.
The goal of this new pipeline is to increase the chance of users clicking into, watching and liking one of the recommended movies, thus improving user retention and activeness on our APP.

The report(reco_final_project.ipynb) include mainly 6 parts:
- In Part 1, a detailed business objective is explicitly explained
- In Part 2, dataset are initially explored and sampled
- In Part 3, two benchmark models (Bias model & Model-based Collaborative Filtering model) are establised
- In Part 4, Approximate Nearest Neighbors (ANN) model is explored
- In Part 5, 2 hybrid models are built
  - Matrix Factorization model + Neural Network model (MF-NN)
  - Approximate Nearest Neighbors model + Neural Network model(ANN-NN)
- In Part 6, Model summary and final recommendation is provided
  
  
  
## structure
```
.  
├── (ml-latest)             # movielens dataset used for this project (dataset is not pushed but should be put in parent directory here)
├── model                   # support modules
   ├── ann_kdtree.py        # Approximate Nearest Neighbors kdtree model
   ├── ann_nn.py            # Approximate Nearest Neighbors model + Neural Network model
   ├── bias_model.py        # bias model
   ├── helper.py            # helper functions
   └── mf_nn.py             # Matrix Factorization model + Neural Network model
├── model_tuning            # model tuning notebooks
   ├── ann_nn_tuning.ipynb  # Approximate Nearest Neighbors model + Neural Network model tuning notebook
   ├── mf_nn_tuning.ipynb   # Approximate Nearest Neighbors model tuning notebook
   ├── ann_tuning.ipynb     # Matrix Factorization model + Neural Network model tuning notebook
   └── mf_tuning.ipynb      # Model-based Collaborative Filtering model tuning
├── reco_final_project      # notebook with recommendation approaches and basic results
├── requirement.txt         # package requirements 
└── README.md               # README file
```

## Requirement
- math
- time
- scipy
- tqdm
- pandas
- numpy
- random
- collections
- matplotlib
- seaborn
- sklearn
- keras
- pyspark>= 3.0.1
