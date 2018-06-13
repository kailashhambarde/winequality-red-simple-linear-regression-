import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

import os 
print (os.getcwd())
os.chdir('C:\Users\Panu\Downloads')

df = pd.read_csv('winequality-red.csv', sep=';')
df.rename(columns=lambda x: x.replace(" ", "_"),inplace=True)
x_train,x_test,y_train,y_test = train_test_split(df['alcohol'], df["quality"],train_size = 0.7,random_state=42)


'''after spliting a single varible out of the DataFrame, it becomes a pandas series hence we need to convert it back into a pandas DataFrame again'''
x_train = pd.DataFrame(x_train);
x_test = pd.DataFrame(x_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

'''the follwoing function is for calculationg the mean from the columns of the DataFrame. The mean was 
calculated for both alcohol(independent) and the quality(dependent) varibles:'''

def mean(values):
    return round(sum(values)/float(len(values)),2)
alcohol_mean = mean(x_train['alcohol'])
quality_mean = mean(y_train['quality'])

'''Variance and covariance is indeed needed for calculationg the coefficient of the regression model'''
alcohol_variance = round(sum((x_train['alcohol']-alcohol_mean)**2),2)
quality_variance = round(sum((y_train['quality']-quality_mean)**2),2)

covariance = round(sum((x_train['alcohol']- alcohol_mean)* (y_train['quality'] - quality_mean)),2)

b1 = covariance/alcohol_variance
b0 = quality_mean - b1*alcohol_mean
print("\n\n Intercept (B0):",round(b0,4),"co-efficient (B1):",round(b1,4))


'''after computing coefficients , it is necessary to predict the quality varible,which will test
quality of fit using R-squared value:'''

y_test["y_pred"] = pd.DataFrame(b0+b1*x_test['alcohol'])
R_sqrd = 1- (sum((y_test['quality'] -y_test['y_pred'])**2)/ sum((y_test['quality'] - mean(y_test['quality']))**2))
print ("test R-squared value", round(R_sqrd,4))
    
    
    
