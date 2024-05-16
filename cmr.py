# Akshar Yeccherla, 2024

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd
from tqdm import tqdm

class CMR:
    """
    Conditional Model Reliance
    Given an imputation model, imputation cross-validation scheme, and cross-validated
    prediction model, calculates the feature importances.

    Parameters
    ----------
    data: data
    labels: labels
    scoring: string or callable
        Same as scoring in sklearn API, regression or classification error
    model: model (sklearn API), optional
        Cross-validated model
    imp_model:
    imp_fit_params : dict, optional
        fit parameters for imputation model
    imp_cv: int or iterable
        Same as cv in sklearn API
    scramble_method: string, optional
      method for scramble of unique info
    n_jobs: int, optional
        Number of jobs for parallel computation
        
    TODO
    -----------
    support categorical and continuous variables
    speed up
    support parallelization (n_jobs)
    support other methods for smaller datasets
    """
    
    def __init__(self, data, labels, scoring, model, imp_model=XGBRegressor(), cv_imputation=False, imp_fit_params=None, imp_cv=5, n_jobs=None):
      self.data = data.to_numpy() if not isinstance(data, np.ndarray) else data
      self.labels = labels.to_numpy() if not isinstance(labels, np.ndarray) else labels
      
      self.model = model
      self.scoring = scoring

      self.imp_model = imp_model
      self.imp_fit_params = imp_fit_params if imp_fit_params else dict()
      self.imp_cv = imp_cv
      self.n_jobs = n_jobs
      self.cv_imputation = cv_imputation

      self.num_observations = np.shape(self.data)[0]
      self.num_features = np.shape(self.data)[1]
      
      self.baseline_score = self.base_score()
    
    def imputation_model_cv(self, data, labels):
      if not self.cv_imputation:
        self.imp_model.fit(data, labels)
        return self.imp_model
      
      gs_cv = RandomizedSearchCV(self.imp_model, param_grid=self.imp_fit_params, cv=self.imp_cv)
      gs_cv.fit(data, labels)
      
      best_model = gs_cv.best_estimator_
      best_model.fit(data, labels)
      
      return best_model
    
    def base_score(self):
      y_pred = self.model.predict(self.data)
      return self.scoring(self.labels, y_pred)
          
    def feature_importance(self, feature):
      feature_mask = [i for i in range(self.num_features) if i != feature]
      data = self.data[:, feature_mask]
      feature_col = self.data[:, feature]
      
      imputation_model = self.imputation_model_cv(data, feature_col)
      
      feature_pred = imputation_model.predict(data)

      feature_unique, feature_impute = self.feature_unique_info(feature_col, feature_pred)
      
      scrambled_score = 0.0
      
      for j in range(self.num_observations):
        data_row = np.copy(self.data[j,:])
        data_label = self.labels[j]
        swap_preds = []
        for i in range(self.num_observations):
          if i == j: 
            continue
          
          feature_ij = feature_unique[i] + feature_impute[j]
          
          data_row[feature] = feature_ij
          
          swap_preds.append(self.model.predict([data_row]))
          
        scrambled_score += self.scoring(swap_preds, np.full(self.num_observations-1, data_label))
                
      scrambled_score = scrambled_score / (self.num_observations)      
      importance = scrambled_score - self.baseline_score
      
      return importance
      
    def feature_unique_info(self, feature, feature_pred):
      # Imputation procedure for categorical, randomized

      # shape = np.shape(feature_pred)
      # rand_num = np.random.rand(shape[0])
      
      # result = feature_pred - rand_num
      # result[result <= 0] = 0
      # result[result > 0] = 1
      
      # feature_unique = feature - result
      
      # return feature_unique, result

      feature_unique = feature - feature_pred
      
      return feature_unique, feature_pred
    
    def importance_all(self, mode="array"):
      print(f"Computing importances for {self.num_features} features")
      if mode == "list":
        importance_list = []
        for i in tqdm(range(self.num_features)):
          importance_list.append((i, self.feature_importance(i)))
        return sorted(importance_list, key=lambda x:(-x[1],x[0]))
      
      importance_list = []
      for i in tqdm(range(self.num_features)):
        importance_list.append(self.feature_importance(i))
        
      return importance_list

