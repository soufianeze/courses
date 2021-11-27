import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def open_df(path):
    df=pd.read_csv(path, sep = ',', header = 0)
    df.head()
    return df

def perform_eda(df, target_column, threshQlyQty = 10, display_charts=False):
    print('-----------------------------------------------')
    print(f'1. TARGET VARIABLE : {target_column}')
    print('-----------------------------------------------')
    print(f'percentage of classes in {target_column}')
    print(df[target_column].value_counts(normalize = True))

    categorical_columns = df.loc[:, df.nunique() < threshQlyQty].columns
    quantitative_columns = df.loc[:, df.nunique() < threshQlyQty].columns


  # (Test Chi2) between target and each quantitative variable
    print('\n\n')
    print('-----------------------------------------------')
    print('(Test Chi2) between target and each quantitative variable')
    print('-----------------------------------------------')
    print('\n\n')

    chi2dict = {}
    for column in categorical_columns:
        table = pd.crosstab(df[target_column], df[column])
        resultats_test = chi2_contingency(table)
        chi2dict[column] = [np.round(resultats_test[0], 2), np.round(resultats_test[1], 2)]
        
    dfChi2 = pd.DataFrame(chi2dict, index=['Statistique de test', 'P_Value'])
    print(dfChi2)


    #if display_charts == True :
    print()
    print('-----------------------------------------------')
    print('Tableau de corrélation :')
    plt.figure(figsize=(8,8))
    sns.heatmap(df[quantitative_columns].corr(),annot=True, cmap="RdBu_r", center =0)
    plt.show()



