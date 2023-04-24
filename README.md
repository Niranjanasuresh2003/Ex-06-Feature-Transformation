# Ex-06-Feature-Transformation

# AIM

To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM

# STEP 1

Read the given Data

# STEP 2

Clean the Data Set using Data Cleaning Process

# STEP 3

Apply Feature Transformation techniques to all the features of the data set

#  STEP 4

Save the data to the file

# CODE
~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
~~~


# OUPUT

# Dataset:

![image](https://user-images.githubusercontent.com/129851738/233903042-e16eee71-6c61-4e46-a90c-a0a502e4bebb.png)


# Head:

![image](https://user-images.githubusercontent.com/129851738/233903115-f65894f5-20b8-4e49-b879-e1cfd36713b9.png)


# Null data:

![image](https://user-images.githubusercontent.com/129851738/233903172-a4629482-0971-4c53-adb2-2db3fe68ff48.png)


# Information:

![image](https://user-images.githubusercontent.com/129851738/233903238-357b9a0f-f6ec-4452-a3fb-5e9067874057.png)


# Description:

![image](https://user-images.githubusercontent.com/129851738/233903309-b0ba0f74-427f-4068-bdf6-d702caa9d798.png)


# Highly Positive Skew:

![image](https://user-images.githubusercontent.com/129851738/233903393-60e09462-5d83-46fd-a357-73b08363f2e9.png)


# Highly Negative Skew:

![image](https://user-images.githubusercontent.com/129851738/233903449-4e942091-5ed7-4c53-9ad9-3b35bf69dcb0.png)


# Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/129851738/233903507-6be00e40-4e34-4eb1-b80d-46d0ee8d965b.png)


# Moderate Negative Skew:

![image](https://user-images.githubusercontent.com/129851738/233903625-745d2ccd-7428-4e7d-9c0c-8041d6961a71.png)


# Log of Highly Positive Skew:

![image](https://user-images.githubusercontent.com/129851738/233903670-e15edd9e-2c1b-4ceb-9e1e-0243927cd979.png)


# Log of Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/129851738/233903728-e6406bae-896a-44a2-8d29-8c10b9cd4d5a.png)

# Reciprocal of Highly Positive Skew:

![image](https://user-images.githubusercontent.com/129851738/233903843-6bfd9746-c0c2-4998-8bae-c2989de70e20.png)

# Square root tranformation:

![image](https://user-images.githubusercontent.com/129851738/233903945-52d95231-1d69-4e07-b62f-12ddf959fae5.png)


# Power transformation of Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/129851738/233904015-997ec218-696b-4c3b-921c-9a9694d89b98.png)


# Power transformation of Moderate Negative Skew:

![image](https://user-images.githubusercontent.com/129851738/233904153-6a464102-bb40-4a9d-9f5b-0cacd5e6e307.png)


#  Quantile transformation:

![image](https://user-images.githubusercontent.com/129851738/233904279-49d41d40-1bbb-4d4b-a7e1-46ae8638026b.png)


# Result

Thus, Feature transformation is performed and executed successfully for the given dataset



