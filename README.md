# Feature_Transformation
# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Print the transformed features

# PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
# OUTPUT:

![1](https://github.com/Krupa-Varsha-P/DS-EX-6/assets/100466625/d292f7d6-dcfb-4823-bf28-514fe8d9dad0)
![2](https://github.com/Krupa-Varsha-P/DS-EX-6/assets/100466625/46abe241-a07e-47e3-9094-1b2fed280386)
![3](https://github.com/Krupa-Varsha-P/DS-EX-6/assets/100466625/1a7c3167-a486-43a5-b0ab-645d3d4dc831)
![4](https://github.com/Krupa-Varsha-P/DS-EX-6/assets/100466625/c62f0ffd-8353-4da1-aadc-0b78d4a0fa5e)
![5](https://github.com/Krupa-Varsha-P/DS-EX-6/assets/100466625/22687568-209a-457b-805c-bfa8053772a7)
![6](https://github.com/Krupa-Varsha-P/DS-EX-6/assets/100466625/aaa6ffb0-8809-4524-8855-bbcdcaf11d1c)
![7](https://github.com/Krupa-Varsha-P/DS-EX-6/assets/100466625/cd8515fa-582d-4b29-933e-07b9f73ce9f2)
![8](https://github.com/Krupa-Varsha-P/DS-EX-6/assets/100466625/aef298fd-60bb-420a-8cc2-46823f5350a3)
![9](https://github.com/Krupa-Varsha-P/DS-EX-6/assets/100466625/f5d624b3-9c77-40e2-a5a5-ad060db79238)
![10](https://github.com/Krupa-Varsha-P/DS-EX-6/assets/100466625/75c8d439-20fb-47e7-b8ab-3eff73baa218)

# RESULT:
Thus feature transformation is done for the given dataset.
