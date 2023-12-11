#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.available
import seaborn as sns

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter("ignore") #ignore warnings for legibility
# interactive visualization
import plotly.express as px
import plotly.graph_objs as go
# import plotly.figure_factory as ff
from plotly.subplots import make_subplots

#Load data Read Data
df = pd.read_csv('COVID-19_Case_Surveillance_Public_Use_Data.csv')
df

#Data Preprocessing
#Shape
#number of rows and columns
print(df.shape)
#data info
df.info
#Understand the data
#First 5 rows
#display the first 5 rows of the data
df.head()
#last 5 rows
df.tail()
#describe data
df.describe()
#Data Visualization
#Confirmed Cases pie chart
values = df['current_status'].value_counts().tolist()
names = ['Confirmed', 'Probable']
fig = px.pie(
 names=names,
 values=values,
 title="Case Status ",

 color_discrete_sequence=px.colors.sequential.RdBu,
)
fig.show()
#Gender Allocation pie chart
values = df['sex'].value_counts().tolist()
names = ['Female', 'Male', 'Unknown', 'Missing', 'Other']
fig = px.pie(
 names=names,
 values=values,
 title="Gender Status",
 color_discrete_sequence=px.colors.sequential.Bluyl_r,
)
fig.show()
#Gender Allocation bar chart
values = df['sex'].value_counts().tolist()
names = ['Female', 'Male', 'Unknown', 'Missing', 'Other']
fig = px.bar(
 x=names,
 y=values,
 title="Gender Status Bar Chart",
 labels={
 'x': 'Status',
 'y': 'Number of Patients'
 },
 color=values
)
fig.show()
#Race and Ethinicity pie chart
values = df['Race and ethnicity (combined)'].value_counts().tolist()
names = ['Unkown', 'White, Non-Hispanic', 'Hispanic/Latino', 'Black, Non-Hispanic', 'Missing',
'Multiple/Other, Non-Hispanic', 'Asian, Non-Hispanic', 'American Indian/Alaska Native, NonHispanic', 'Native Hawaiian/Other Pacific Islancer, Non-Hispanic']
fig = px.pie(
 names=names,
 values=values,
 title="Races and Ethinicities ",
 color_discrete_sequence=px.colors.sequential.Electric,
)
fig.show()

#Data inspection and cleaning
#Null values
df.info(null_counts=True)
#round Null values
round(100*(df.isnull().sum()/len(df.index)), 10)
#drop values
df = df.drop(['pos_spec_dt','onset_dt'],axis=1)
df.info()
#round null values
round(100*(df.isnull().sum()/len(df.index)), 2)
#new first 5 rows
df.head()
#Covid19 current status
df['current_status'].value_counts()
#value count sex
df['sex'] = df['sex'].apply(lambda x: 'Unknown' if x == 'Missing' else x)
df['sex'].value_counts()
#value count race and ethicity
df['Race and ethnicity (combined)'] = df['Race and ethnicity (combined)'].apply(lambda x:
'Unknown' if x == 'Missing' else x)
df['Race and ethnicity (combined)'].value_counts()
#value count hosp_yn
df['hosp_yn'] = df['hosp_yn'].apply(lambda x: 'Unknown' if x == 'Missing' else x)
df.hosp_yn.value_counts()
#value count deaths
df['death_yn'] = df['death_yn'].apply(lambda x: 'Unknown' if x == 'Missing' else x)
df.death_yn.value_counts()
#value count medcond_yb
df['medcond_yn'] = df['medcond_yn'].apply(lambda x: 'Unknown' if x == 'Missing' else x)
df.medcond_yn.value_counts()
#cdc report first 5 values

df.cdc_report_dt.head()
#dates split
df['Year'] = df.cdc_report_dt.apply(lambda x: int(x.split('/')[0]))
df['Month'] = df.cdc_report_dt.apply(lambda x: int(x.split('/')[1]))
df['Date'] = df.cdc_report_dt.apply(lambda x: int(x.split('/')[2]))
#drop cdc report date
df = df.drop('cdc_report_dt',axis=1)
#first 5 values after dropping cdc repot date
df.head()
#data info after dropping cdc repot date
df.info()
#value count age group
df['age_group'].value_counts()
#Dropping all rows with 'Unknown values' as well as dropping columns that are extremely skewed.
df = df[df['Race and ethnicity (combined)'] != 'Unknown']
df = df[df.death_yn != 'Unknown']
df = df[df['hosp_yn'] != 'Unknown']
df = df.drop('icu_yn',axis=1)
df = df.drop('medcond_yn',axis=1)
df = df.drop('Year',axis=1)
df = df[df.sex != 'Unknown']
df = df[df.age_group != 'Unknown']
df.head()
#value count per month
lis = ['Jan','Feb','March','April','May','June','July','Aug','Sept','Oct','Nov','Dec']
df['Month'] = df['Month'].apply(lambda x : lis[x-1])
df['Month'].value_counts()
#data inspection and cleaning completed
df.head()
#EDA: Exploratory Data Analysis
#plots
# death vs age group
plt.figure(figsize=(12,5))
sns.countplot(x=df['age_group'],hue=df['death_yn'])
plt.title('Death vs Age Group')
plt.xlabel('Age Group')
plt.show()
#age group vs hospitalisation
plt.figure(figsize=(12,5))
sns.countplot(x=df['age_group'],hue=df['hosp_yn'])
plt.title('Hospitalisation vs Age Group')
plt.xlabel('Age Group')
plt.show()
#hosp_yn vs current status
plt.figure(figsize=(12,5))
sns.countplot(x=df['current_status'],hue=df['hosp_yn'])
plt.title('Hospitalisation vs Current Status')
plt.xlabel('Current Status')
plt.show()
plt.figure(figsize=(12,5))
sns.countplot(x=df['current_status'],hue=df['death_yn'])
plt.title('Death vs Current Status')
plt.xlabel('Current Status')
plt.show()
plt.figure(figsize=(12,5))
sns.countplot(y=df['Race and ethnicity (combined)'],hue=df['hosp_yn'])
plt.title('Hospitalisation vs Race and ethnicity (combined)')
plt.ylabel('Race and ethnicity (combined)')
plt.show()
plt.figure(figsize=(12,5))
sns.countplot(y=df['Race and ethnicity (combined)'],hue=df['death_yn'])
plt.title('Death vs Race and ethnicity (combined)')
plt.ylabel('Race and ethnicity (combined)')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='Month',hue='age_group',data=df)
plt.title('Month vs Age Group')
plt.ylabel('Month') 
plt.legend(loc='upper right')
plt.show()
plt.figure(figsize=(12,5))
sns.barplot(x='age_group',y='Date',data=df)
plt.title('Date vs Age Group')
plt.ylabel('Date')
plt.xlabel('Age Group')
plt.show()
plt.figure(figsize=(12,5))
sns.countplot(x='Month',hue='death_yn',data=df)
plt.title('Death vs Month')
plt.xlabel('Month')
plt.legend(loc='upper right')
plt.show()
plt.figure(figsize=(10,5))
sns.barplot(x='death_yn',y='Date',data=df)
plt.title('Date vs Death')
plt.xlabel('Death')
plt.show()
#Building Models
#Relationship between the target and dependent variables.
#Dummy variables creation
#First dummy not dropped
status = pd.get_dummies(df['current_status'],drop_first=False)
df = pd.concat([df, status], axis = 1)
df.drop(['current_status'], axis = 1, inplace = True)
status = pd.get_dummies(df['age_group'], drop_first = False)
df = pd.concat([df, status], axis = 1)
df.drop(['age_group'], axis = 1, inplace = True)
status = pd.get_dummies(df['Race and ethnicity (combined)'], drop_first = False)
df = pd.concat([df, status], axis = 1)
df.drop(['Race and ethnicity (combined)'], axis = 1, inplace = True)
status = pd.get_dummies(df['sex'], drop_first = False)
df = pd.concat([df, status], axis = 1)
df.drop(['sex'], axis = 1, inplace = True)
status = pd.get_dummies(df['Month'], drop_first = False)
df = pd.concat([df, status], axis = 1)

df.drop(['Month'], axis = 1, inplace = True)
df.head()
#Value count death, hosp
df['death_yn'] = df['death_yn'].map({'Yes': 1, "No": 0})
df['hosp_yn'] = df['hosp_yn'].map({'Yes': 1, "No": 0})
df.death_yn.value_counts()
df.info()
#dropping death_yn
X = df.drop(['death_yn'], axis=1)
X.head()
#training and testing data
#due to the large size of the dataset we will use train data 1% vs test data 99%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05, random_state=100)
X_train.head()
#RandomForest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier(random_state=100,n_jobs=-1,class_weight='balanced')
params = {'n_estimators':[100], 
     'min_samples_leaf':[30,60,80,100,200,400,600,800,1000],
 'max_depth':[2,3,5,10,15,30,40,50],
 'max_features':[0.1,0.2,0.3,0.4,0.5],
 'criterion':["gini","entropy"]}
grid_search = GridSearchCV(estimator=model,param_grid=params,verbose=1,n_jobs=-1,scoring='accuracy')
grid_search.fit(X_train,y_train)
#best model using best_estimator
bestmodel = grid_search.best_estimator_
# checking performance of the model
#accuracy score of scoring model train
#accuracy score train
from sklearn.metrics import roc_curve, accuracy_score, recall_score

from sklearn import metrics
y_train_pred = bestmodel.predict(X_train)
print("Accuracy: ", accuracy_score(y_train, y_train_pred))
print("Recall: ", recall_score(y_train, y_train_pred))
#roc curve1 scoring model 
         from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(y_train, y_train_pred)
auc = auc(fpr, tpr)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(fpr, tpr, linestyle='-',label='roccurve(auc = %0.3f)'%auc)
#accuracy score, recall score test
y_test_pred = bestmodel.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_test_pred))
print("Recall: ", recall_score(y_test, y_test_pred))
#roc curve2 recall score
from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(y_test, y_test_pred)
auc = auc(fpr, tpr)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(fpr, tpr, linestyle='-',label='roccurve(auc = %0.3f)'%auc)
#features importance levels
Feature_Importance =
pd.DataFrame({'Feature':X_train.columns,'Importance':bestmodel.feature_importances_})
Feature_Importance.sort_values(by='Importance',ascending=False,inplace=True)
Feature_Importance.set_index('Feature',inplace=True)
Feature_Importance
#Scoring modelGridSearchCV #comparison model
model = RandomForestClassifier(random_state=100,n_jobs=-1,class_weight='balanced')
params = {'n_estimators':[100],
 'min_samples_leaf':[30,60,80,100,200,400,600,800,1000],
 'max_depth':[2,3,5,10,15,30,40,50],
 'max_features':[0.1,0.2,0.3,0.4,0.5],
 'criterion':["gini","entropy"]}
grid_search = GridSearchCV(estimator=model,param_grid=params,verbose=1,n_jobs=-1,scoring='recall')
grid_search.fit(X_train,y_train)

#best model
bestmodel = grid_search.best_estimator_
#accuracy score of scoring model train
y_train_pred = bestmodel.predict(X_train)
print("Accuracy: ", accuracy_score(y_train, y_train_pred))
print("Recall: ", recall_score(y_train, y_train_pred))
#roc curve3 scoring model
from sklearn.metrics import roc_curve, auc
                  fpr, tpr, threshold = roc_curve(y_test, y_test_pred)
auc = auc(fpr, tpr)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(fpr, tpr, linestyle='-',label='roccurve(auc = %0.3f)'%auc)
#accuracy score test
y_test_pred = bestmodel.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_test_pred))
print("Recall: ", recall_score(y_test, y_test_pred))
#roc curve4 score test
from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(y_test, y_test_pred)
auc = auc(fpr, tpr)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(fpr, tpr, linestyle='-',label='roccurve(auc = %0.3f)'%auc)
#security Bandit and Safety tool that checks for security in codes
#always save a file before uploading it. Do not use tokens
#file saved in temp_dir could cause security issues
# bandit package installed
import os
path = os.getcwd()
filename = 'COVID-19_Case_Surveillance_Public_Use_Data.csv'
filepath = os.path.join(path, filename)
print(filepath)
#install bandit
pip install bandit

#Security Checks from cmd window
bandit -r C:\\Users\\nelly\\COVID-19_Case_Surveillance_Public_Use_Data.csv 



                  
