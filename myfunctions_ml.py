import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def train_and_evaluate(model, X_train, X_test, y_train, y_test, list_of_metrics):
  '''
  que fait la fonction

  model : c'est un model de sklearn !
  X_train, X_test : tableau numpy
  list_of_metrics : une liste de métriques PROVENANT de sklearn.metrics (c'est nécessaire)
  '''

  
  # entrainement du modele
  # print(f'{model.__class__.__name__}')
  model.fit(X_train, y_train)

  # le dictionnaire qui va contenir les scores
  dict_of_scores = {}

  # evaluate model on Train_set on all metrics
  # print('\n\n-----------------------------------------------')
  # print('EVALUATE ON TRAIN')
  # print('-----------------------------------------------')
  y_pred = model.predict(X_train)
  for metric in list_of_metrics:
    # print(f'{metric.__name__} :  {np.round(metric(y_train, y_pred), 2)}')
    dict_of_scores[f'TRAIN_{metric.__name__}'] = np.round(metric(y_train, y_pred), 2)


  # evaluate model on Test_set on all metrics
  # print('\n\n-----------------------------------------------')
  # print('EVALUATE ON TEST')
  # print('-----------------------------------------------')
  y_pred = model.predict(X_test)
  for metric in list_of_metrics:
    # print(f'{metric.__name__} :  {np.round(metric(y_test, y_pred), 2)}')
    dict_of_scores[f'TEST_{metric.__name__}'] = np.round(metric(y_test, y_pred), 2)

  # print('-----------------------------------------------')
  # print('-----------------------------------------------\n\n')
  return pd.Series(dict_of_scores)

def compare_models(list_of_models, X_train, X_test, y_train, y_test, list_of_metrics):
  '''
  que fait cette fonction
  '''

  list_of_scores = []
  for model in list_of_models:
    scores = train_and_evaluate(model, X_train, X_test, y_train, y_test, list_of_metrics)
    scores.name = model.__class__.__name__
    list_of_scores.append(scores)

  df = pd.concat(list_of_scores, axis=1)
  df = df.T
  df = df.sort_values(by='TEST_f1_score', ascending=False)
  return df

