import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/Churn_Modelling.csv")
print(df[df['Exited']==1].shape)
print(df[df['Exited']==0].shape)
print(df['Geography'].unique())
df.tail()

encoder = LabelEncoder()
df['Surname']=encoder.fit_transform(df['Surname'])
df['Geography']=encoder.fit_transform(df['Geography'])
df['Gender']=encoder.fit_transform(df['Gender'])
df.head()

x=df.drop(columns=['RowNumber', 'Exited'])
y=df['Exited']

parameters = {'C':[0.01, 0.1, 1, 10, 100]}
model_LR = make_pipeline(GridSearchCV(estimator=LogisticRegression(),param_grid=parameters,cv=2,verbose=2,n_jobs=-1))


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model_LR.fit(x_train,y_train)
pred = model_LR.predict(x_test)
print(classification_report(y_test,pred))

cm=confusion_matrix(y_test,pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham','Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Spam Classification (with Oversampling)')
plt.show()

print(accuracy_score(y_test,pred))
print(f1_score(y_test,pred))

ros = RandomOverSampler(random_state=42)
x_resampled, y_resampled = ros.fit_resample(x_train,y_train)
x_resampled.shape,y_resampled.shape

parameters = {'C':[0.01, 0.1, 1, 10, 100]}
model_LR = make_pipeline(GridSearchCV(estimator=LogisticRegression(),param_grid=parameters,cv=2,verbose=2,n_jobs=-1))

model_LR.fit(x_resampled,y_resampled)
pred = model_LR.predict(x_test)
print(classification_report(y_test,pred))
news = pd.DataFrame({'CustomerId':[145345344],	'Surname':[1132], 'CreditScore':	[722], 'Geography':	[0], 'Gender':	[0],'Age':	[23],'Tenure':	[2],'Balance':	[5.00],'NumOfProducts':	[1],	'HasCrCard':[1],	'IsActiveMember':[1], 'EstimatedSalary':	[1034548.88]})
print("result:",model_LR.predict(news))

cm=confusion_matrix(y_test,pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham','Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Spam Classification (with Oversampling)')
plt.show()

print(accuracy_score(y_test,pred))
print(f1_score(y_test,pred))
