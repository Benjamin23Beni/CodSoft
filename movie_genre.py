from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

cols1=["id","Director","Genre","Summary"]
cols2=["id","Director","Summary","Genre"]
train_df = pd.read_csv("data/Genre Classification Dataset/train_data.txt",names=cols1, sep=':::', engine='python')
test_df = pd.read_csv("data/Genre Classification Dataset/test_data.txt",names=cols2, sep=':::', engine='python')
ans_df = pd.read_csv("data/Genre Classification Dataset/test_data_solution.txt",names=cols1, sep=':::', engine='python')
pd.set_option('display.max_colwidth', None)
print("Duplicated:",train_df.duplicated().sum())
print("Null:\n",train_df.isnull().sum())

parameters = {'C':[0.01, 0.1, 1, 10, 100]}
model_SVC = make_pipeline(TfidfVectorizer(),GridSearchCV(estimator=LinearSVC(),param_grid=parameters,cv=2,verbose=2,n_jobs=-1))

parameters = {'C':[0.01, 0.1, 1, 10, 100]}
model_LR = make_pipeline(TfidfVectorizer(),GridSearchCV(estimator=LogisticRegression(),param_grid=parameters,cv=2,verbose=2,n_jobs=-1))

parameters = {'alpha':[0.01, 0.1, 1, 10, 100]}
model_NB = make_pipeline(TfidfVectorizer(),GridSearchCV(estimator=MultinomialNB(),param_grid=parameters,cv=2,verbose=2,n_jobs=-1))

model = VotingClassifier(estimators=[('svc', model_SVC), ('lt', model_LR), ('nb', model_NB)], voting='hard')
model.fit(train_df['Summary'], train_df['Genre'])
model.predict(test_df['Summary'])
print("Classification report: ",classification_report(ans_df['Genre'], model.predict(test_df['Summary'])))
