# ODD2023-Datascience
# Ex-06-Feature-Transformation
## AIM :

To read the given data and perform Feature Transformation process and save the data to a file.
## EXPLANATION :

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
## ALGORITHM :
### STEP 1:

Read the given Data
### STEP 2:

Clean the Data Set using Data Cleaning Process
### STEP 3:

Apply Feature Transformation techniques to all the features of the data set
### STEP 4:

Print the transformed features
## PROGRAM :
```
DEVELOPED BY:YOGABHARATHI.S
REGISTER NO:212222230179
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
## OUTPUT:
### data:
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/81a9eb4b-c2bb-4f93-b7ae-651aa93d18ca)

### data.head():
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/fe06fc5c-9573-4f3f-b573-bdcb38f89cb6)

### data.info():
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/6616a126-0a96-4975-99da-47718227238f)

### df.describe():
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/a2a6fb9c-9326-48f3-b7f4-853a46d74c13)

### df.isnull().sum():
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/89627d5e-3658-4e5f-9bcd-6bd3c380e76b)

### Before transformation:
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/e80d08bd-8eea-46f5-9683-6c7ba705f6d0)
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/affa87e2-ecdd-4df5-a744-d9e7f8f80ae1)
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/b49108c1-3d8c-4f23-bd5b-590467fb1082)
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/bad4f41f-9149-4503-a654-508eae792fc6)

### Log transformation :
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/0bc56551-363e-4693-82b5-7d67e05a9a98)

### Reciprocal tranformation:
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/cce6bcca-afd0-4398-95ef-f903fb7ca8d9)

### Square root transformation:
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/024ed22b-502e-41b1-840d-0905775f3c34)
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/b13dd43f-a51f-47c8-8f69-647bb01064c7)
### Power transformation:
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/12b5cb97-2476-4eca-82af-a9a31c88a3ac)

### Quantile transformation:
![image](https://github.com/Yogabharathi3/ODD2023-Datascience-Ex06/assets/118899387/4518c1f4-8478-4915-a55f-adf70b892e0f)

## RESULT:
Thus, Feature transformation is performed and executed successfully for the given dataset.
