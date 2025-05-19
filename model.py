# SVM

import numpy as np


from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA


# read pickle file
import pandas as pd
import os
from tqdm import tqdm


df = pd.read_pickle('features_nolpass.pkl')

# split the data into train and test
X = []
y = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    X.append(row['feature_vector'])
    y.append(row['activity'])

X = np.array(X)
X = X[:, :]
y = np.array(y)
# convert y to numerical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


print(X.shape)


print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.0001,
    objective='multi:softmax',
    num_class=len(df['activity'].unique()),
    random_state=42
)
#svm.SVC(kernel='linear', C=1.0, random_state=42,               )
""""""
#
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)



print(classification_report(y_test, y_pred))
print(classification_report(y_train, clf.predict(X_train))) # To check if the model is overfitting or not

"""# plot confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns


confusion_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
labels = df['activity'].unique()

sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)


plt.xlabel('Predicted')
plt.ylabel('True')  
plt.title('Confusion Matrix')
plt.show()"""