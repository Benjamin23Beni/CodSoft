import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import date
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,f1_score


train_df=pd.read_csv("data/fraudTrain.csv")
test_df=pd.read_csv("data/fraudTest.csv")
print(train_df.shape)
print(train_df[train_df['is_fraud']==0].shape)
print(train_df[train_df['is_fraud']==1].shape)
outliers=len(train_df[train_df['is_fraud']==0])/len(test_df[test_df['is_fraud']==1])
print("Outliers: ",outliers)
train_df.head()

train_df['trans_date_trans_time']=pd.to_datetime(train_df['trans_date_trans_time'])
train_df['dob']=pd.to_datetime(train_df['dob'])
train_df['age']=(pd.to_datetime(date.today())-train_df['dob']).dt.days / 365.25
train_df['distance']=(train_df['lat']-train_df['merch_lat'])**2+(train_df['long']-train_df['merch_long'])**2
pd.set_option('display.max_colwidth', None)

encoder = LabelEncoder()
train_df['gender']=encoder.fit_transform(train_df['gender'])
train_df['category']=encoder.fit_transform(train_df['category'])
train_df['merchant']=encoder.fit_transform(train_df['merchant'])
train_df['street']=encoder.fit_transform(train_df['street'])
train_df['city']=encoder.fit_transform(train_df['city'])
train_df['state']=encoder.fit_transform(train_df['state'])
train_df['job']=encoder.fit_transform(train_df['job'])

train_df['trans_date_trans_time'] = train_df['trans_date_trans_time'].astype(int)
train_df['dob'] = train_df['dob'].astype(int)

x=train_df.drop(columns=['is_fraud','cc_num','first','last','trans_num'])
y=train_df['is_fraud']

vector = TfidfVectorizer()
vectors = vector.fit(x,y)
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x_train,y_train)

parameters = {'C':[0.01, 0.1, 1, 10, 100]}
model_LR = GridSearchCV(estimator=LogisticRegression(),param_grid=parameters,cv=2,verbose=2,n_jobs=-1)

parameters = {'max_depth': range(1, 10, 1),
    'min_samples_leaf': range(1, 20, 2),
    'min_samples_split': range(2, 20, 2),
    'criterion': ["entropy", "gini"]
}
model_DT = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),param_grid=parameters,cv=2,verbose=2,n_jobs=-1)

parameters = {'max_features': ['sqrt','log2']}
model_RF = GridSearchCV(estimator=RandomForestClassifier(),param_grid=parameters,cv=2,verbose=2,n_jobs=-1)

model = VotingClassifier(estimators=[('svc', model_LR)], voting='soft')
model.fit(x_resampled, y_resampled)
y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham','Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Spam Classification (with Oversampling)')
plt.show()

