# packages for processing data
import pandas as pd
import numpy as np
import math

# packages for VIF calculations
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from scipy import stats



# MAD based outlier elimination
def mad_based_outlier(points, thresh=3.5):
    '''
    thresh : using standard deviation, this value dictates how much pecentage will be eliminated(default: 3.5%)
    points : np.series
    return : returns if the value of each row is an outlier in True/False in series.
    '''
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    # MAD (Median Absolute Deviation)
    # 0.6745 is the 0.75th quartile of the standard normal distribution, to which the MAD converges to.
    # https://medium.com/james-blogs/outliers-make-us-go-mad-univariate-outlier-detection-b3a72f1ea8c7
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh 
# reference: https://pythonanalysis.tistory.com/7 [Python Data Analysis]



# Functionto check Variance Inflation Factor(VIF) to minimize Multicollinearity
def lowVIF(df, n=7, cols =['temp', 'cloud', 'wind','humid', 'hpa', 'sun_time', 'lgt_time', 
       'SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25'] ):
    '''
    cols_using : put names of column, in array, which you want to caculate VIF
                by default, column names that relates to the project is set
    '''
    col_to_use = cols
    vif = pd.DataFrame()
    vif["VIF_Factor"] = [variance_inflation_factor(
        df[col_to_use].values, i) for i in range(df[col_to_use].shape[1])]
    vif["features"] = col_to_use
    vif.sort_values("VIF_Factor")
    lowest_vif = vif.sort_values("VIF_Factor")[:n].reset_index()
    lowest_vif.drop(columns='index', inplace=True)
    # returns in df. 0: column name, 1: value of VIF
    return lowest_vif

#########################################################################
# function to process modeling 
def linReg(df, item, cols):
    cols_using = cols
    X = df.loc[df['category']==item, cols_using]
    y = df.loc[df['category']==item,'qty']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

    model = LinearRegression().fit(X_train, y_train)
  
    print('Analysis result of %s using LinearRegression :'%item)
    print('Training Set Score : {:.2f}'.format(model.score(X_train, y_train)))
    print('Verificaiton Set Score : {:.2f}'.format(model.score(X_test, y_test)))

    
def ridgeReg(df, item, cols_using):
    cols = cols_using
    X = df.loc[df['category']==item,cols]
    y = df.loc[df['category']==item,'qty']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

    ridge = Ridge(alpha=0.1, normalize=True, random_state=0, tol=0.001).fit(X_train, y_train)
    
    print('Analysis result of %s using RidgeRegression :'%item)
    print('Training Set Score : {:.2f}'.format(ridge.score(X_train, y_train)))
    print('Verificaiton Set Score : {:.2f}'.format(ridge.score(X_test, y_test)))


def lassoReg(df, item, cols_using):
    cols = cols_using
    X = df.loc[df['category']==item,cols]
    y = df.loc[df['category']==item,'qty']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

    lasso = Lasso(alpha=0.1, max_iter=1000).fit(X=X_train, y=y_train)
  
    print('Analysis result of %s using LassoRegression :'%item)
    print('Training Set Score : {:.2f}'.format(lasso.score(X_train, y_train)) )
    print('Verificaiton Set Score : {:.2f}'.format(lasso.score(X_test, y_test)) )

    #Number of used properties
    print('Number of used properties : {}'.format(np.sum(lasso.coef_ != 0)) )

#########################################################################
# Function to merge datagrames, 'item' specifies which Retail products will be used and 
# 'on_what' decides on a which column should dataframes merged
def mergeForAnalysis(df1, df2, df3, item, on_what='date'):
    merged_df = pd.merge(df1.loc[df1.category==item], df2, on=on_what, how='left')
    merged_df = pd.merge(merged_df, df3, on=on_what, how='left')
    return merged_df

#########################################################################
# creating formula for Ordinary Least Square(ols)
def formulaGen(target, ind_features):
    '''
    formulaGen(target_column_name,[independent_feature_column1, independent_feature_column2,...])
    will return str
    '''
    custom_formula = target + " ~ "
    for f in range(len(ind_features)):
        custom_formula += ind_features[f]
        if f!=(len(ind_features)-1):
            custom_formula += " + "
    return custom_formula
#########################################################################
