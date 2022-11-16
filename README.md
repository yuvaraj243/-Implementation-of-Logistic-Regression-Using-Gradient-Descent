# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the given dataset.
2. Fitting the dataset into the training set and test set.
3. Applying the feature scaling method.
4. Fitting the logistic regression into the training set.
5. Prediction of the test and result
6. Making the confusion matrix
7.Visualizing the training set results.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: YUVARAJ.V
RegisterNumber:  212220220056
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets = pd.read_csv("/content/Social_Network_Ads (1).csv")
X = datasets.iloc[:,[2,3]].values
Y = datasets.iloc[:,[4]].values

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X

X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.fit_transform(X_Test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_Train, Y_Train)

Y_Pred = classifier.predict(X_Test)
Y_Pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)
cm

from sklearn import metrics
accuracy = metrics.accuracy_score(Y_Test, Y_Pred)
accuracy

recall_sensitivity = metrics.recall_score(Y_Test, Y_Pred, pos_label = 1)
recall_specificity = metrics.recall_score(Y_Test, Y_Pred, pos_label = 0)
recall_sensitivity, recall_specificity

from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1,X2 = np.meshgrid(np.arange(start = X_Set[:,0].min()-1, stop = X_Set[:,0].max()+1, step = 0.01), 
                    np.arange(start = X_Set[:,1].min()-1, stop = X_Set[:,1].max()+1, step = 0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),
X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(), X2.max())
plt.ylim(X1.min(), X2.max())
for i,j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j,0],X_Set[Y_Set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.label('Estimated Salary')
plt.legend()
plt.show()

```

## Output:

## prediction of test result:
![ss1](https://user-images.githubusercontent.com/115924983/196043905-486ae8b3-8331-497f-b56c-31c043d5534a.jpg)

## Confusion Matrix:
![ss2](https://user-images.githubusercontent.com/115924983/196043950-f2820ad1-5587-4f10-906a-37fbbc534063.jpg)

## Accuracy:
![ss3](https://user-images.githubusercontent.com/115924983/196043995-f2e01dc1-9b4a-4531-91d4-78297fce9805.jpg)

## Recalling Sensitivity and Specificity:
![ss4](https://user-images.githubusercontent.com/115924983/196044010-c9d30091-f1ae-48d7-b6eb-ac7eda2d96ac.jpg)

## Visulaizing Training set Result:
![ss5](https://user-images.githubusercontent.com/115924983/196044026-87238184-3303-445b-8f37-ba7698eae18e.png)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

