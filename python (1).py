# **Installing Libraries**
# pip install seaborn
# pip install numpy
# pip install pandas
# pip install matplotlib
# pip install seaborn

# importing libraries
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# Load the data
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('data.csv')
df.head()

# Number and rows and columns in the datasets
df.shape

# Count the number of empty values in each column
df.isna().sum()

# Drop the empty column
df = df.dropna(axis=1)

# Number and rows and columns in the updates datasets(without empty values)
df.shape

# Using Label Encoder to label the categorical data
labelencoder_Y = LabelEncoder()
df.iloc[:, 1] = labelencoder_Y.fit_transform(df.iloc[:, 1].values)

# Get the correlation of the columns
df.iloc[:, 1:32].corr()

# visualize the correlation using figure
plt.figure(figsize=(10, 10))
sns.heatmap(df.iloc[:, 1:32].corr(), annot=True, fmt='.0%')

# split the data set into independent (x) and dependent (y) data sets and converting it into array
X = df.iloc[:, 2:31].values
Y = df.iloc[:, 1].values

# split the data set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Function for Models


def models(X_train, Y_train):

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)

    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(
        n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, Y_train)

    # Printing the Training Data Accuracy
    print('[0]Logistic Regression trainning accuracy:',
          log.score(X_train, Y_train))
    print('[1]Decision trainning accuracy:', tree.score(X_train, Y_train))
    print('[2]Random Forest classifier trainning accuracy:',
          forest.score(X_train, Y_train))

    return log, tree, forest


model = models(X_train, Y_train)

# testing the model on the test data (confusion matrix)
for i in range(len(model)):
    print('Model ', i)
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    print(cm)
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    Accuracy = (TP + TN)/(TP+TN+FN+FP)
    print('Accuracy of model ', Accuracy)
