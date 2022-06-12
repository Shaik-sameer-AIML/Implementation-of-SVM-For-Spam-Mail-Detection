# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.
5.End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Shaik sameer 
RegisterNumber:  212221240051
*/


import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Data Head:
![1](https://user-images.githubusercontent.com/93427186/173214244-bfa85bc1-14c3-4cb6-906a-d38d903d22f6.PNG)
### Data Info:
![2](https://user-images.githubusercontent.com/93427186/173214261-8f4fd230-daa1-44c6-984a-8eb577c19eb0.PNG)
### Data isnull():
![3](https://user-images.githubusercontent.com/93427186/173214278-3547a1e6-d6b0-4e91-87a6-8f2d5322ed42.PNG)
### y_pred:
![4](https://user-images.githubusercontent.com/93427186/173214294-e1e9f860-7ca7-4c9f-806b-d26d36fd9538.PNG)
### Accuracy:
![5](https://user-images.githubusercontent.com/93427186/173214305-8e4be995-45ab-4231-925e-c562fc6d455d.PNG)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
