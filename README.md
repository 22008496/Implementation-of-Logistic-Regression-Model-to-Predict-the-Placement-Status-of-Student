# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MAGESH N
RegisterNumber:  212222040091
*/

import pandas as pd
data=pd.read_csv("/content/Placement_Data (1).csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()
data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



```

## Output:

placement data:

![Screenshot 2023-10-02 134240](https://github.com/22008496/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476113/bf474ed9-4a90-4e4d-9fb8-69d6be0e3056)

salary data:

![Screenshot 2023-10-02 134254](https://github.com/22008496/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476113/50e41841-a879-487e-8351-62facfc55b2e)

checking the null() function:

![Screenshot 2023-10-02 134328](https://github.com/22008496/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476113/ee49053e-50c4-4720-9584-c8cc5a7197d9)

data duplicate:

![Screenshot 2023-10-02 134408](https://github.com/22008496/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476113/8206f9d9-15c2-47b3-a32e-5ef162744971)

print data:

![Screenshot 2023-10-02 134425](https://github.com/22008496/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476113/da53b860-0666-478e-9834-a90559fc026c)

data status:

![Screenshot 2023-10-02 134439](https://github.com/22008496/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476113/36214a03-7e8b-4018-a9c9-4f29e02e0530)

y_precidiction array:

![Screenshot 2023-10-02 134448](https://github.com/22008496/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476113/aabf9bc6-9aa4-46b0-b00d-6035f0ab14b1)

accuracy value:

![Screenshot 2023-10-02 134500](https://github.com/22008496/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476113/141703d7-86e8-4a26-9239-45f4a44439ec)

prediction of LR:

![Screenshot 2023-10-02 134520](https://github.com/22008496/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476113/1114b5b2-ec0d-44b8-9b0c-c142314da445)

precidicted of LR:

![Screenshot 2023-10-02 134545](https://github.com/22008496/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119476113/7e2c9992-5a20-42cc-85ce-ab3a2d03f06f)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
