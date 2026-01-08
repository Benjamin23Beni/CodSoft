import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split

cols=['type','message',"null1","null2","null3"]
df=pd.read_csv("data/spam.csv", encoding='latin-1',names=cols)
print(df.shape)
df=df.drop(0)
print(df['type'].isnull().sum())
print(df['message'].isnull().sum())
print(df['null1'].isnull().sum())
print(df['null2'].isnull().sum())
print(df['null3'].isnull().sum())
x=df['message']
y=df['type']
print("spam",df[df['type']=='spam'].shape)
print("ham",df[df['type']=='ham'].shape)
outliers=len(df[df['type']=='ham'])/len(df[df['type']=='spam'])
print("Outliers:",outliers)
(df.head())


type_counts = y.value_counts()
plt.figure(figsize=(7, 5))
type_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Distribution of Ham vs Spam Messages')
plt.xlabel('Message Type')
plt.ylabel('Number of Messages')
plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
pipe = make_pipeline(TfidfVectorizer(), SVC(C=1,kernel='linear',class_weight='balanced'))
pipe.fit(x_train, y_train)
prediction=pipe.predict(x_test)
print(classification_report(y_test,prediction))
print("Predicted: \n",prediction[10:])

cm=confusion_matrix(y_test,prediction)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham','Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Spam Classification (with Oversampling)')
plt.show()


vectors = TfidfVectorizer()
x_vectorized=vectors.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_vectorized,y,test_size=0.2,random_state=42)
ada = ADASYN(random_state=42)
x_resampled, y_resampled = ada.fit_resample(x_train,y_train)

type_counts = y_resampled.value_counts()
plt.figure(figsize=(7, 5))
type_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Distribution of Ham vs Spam Messages')
plt.xlabel('Message Type')
plt.ylabel('Number of Messages')
plt.show()

model = SVC(class_weight='balanced')
model.fit(x_resampled, y_resampled)
prediction=model.predict(x_test)
print(classification_report(y_test,prediction))

cm=confusion_matrix(y_test,prediction)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham','Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Spam Classification (with Oversampling)')
plt.show()
