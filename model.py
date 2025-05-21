# SVM

import numpy as np


from sklearn import svm
from sklearn.calibration import label_binarize
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score


# read pickle file
import pandas as pd
import os
from tqdm import tqdm


df = pd.read_pickle('features_hog_hof.pkl')

# split the data into train and test
X = []
y = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    #if row['activity'] == 'handwaving' or row['activity'] == 'jogging':
     #   continue
    X.append(row['feature_vector'])
    y.append(row['activity'])

X = np.array(X)
X = X[:, :]
y = np.array(y)
# convert y to numerical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)



X = PCA(n_components=0.8).fit_transform(X)

print(X.shape)


print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True)           
              

#
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)



print(classification_report(y_test, y_pred))
print(classification_report(y_train, clf.predict(X_train))) # To check if the model is overfitting or not



# give me the following balanced metric balanced_score = (f1_macro + f1_weighted + average_auc + average_precision) / 4

# F1 scores
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

# Binarize labels for multiclass AUC and average precision
y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))
y_pred_proba = clf.predict_proba(X_test)

# Compute average AUC (macro)
try:
    average_auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
except ValueError:
    average_auc = np.nan  # In case only one class present in y_test

# Compute average precision (macro)
average_precision = average_precision_score(y_test_bin, y_pred_proba, average='macro')

balanced_score = np.nanmean([f1_macro, f1_weighted, average_auc, average_precision])

print("Balanced Score: ", balanced_score)



# plot confusion matrix


confusion_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))

# plot normalized confusion matrix
confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(confusion_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False,
            xticklabels=le.classes_, yticklabels=le.classes_)


plt.xlabel('Predicted')
plt.ylabel('True')  
plt.title('Confusion Matrix')
plt.show()




fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(le.classes_)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 7))
colors = plt.cm.get_cmap('tab10', n_classes)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
             label=f'ROC curve of class {le.classes_[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# plot precision recall curve

precision = dict()
recall = dict()
avg_precision = dict()

plt.figure(figsize=(10, 7))
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
    avg_precision[i] = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(recall[i], precision[i], lw=2,
             label=f'PR curve of class {le.classes_[i]} (AP = {avg_precision[i]:0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()



