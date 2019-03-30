#Importing the Libraries
 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
          
#Load the dataset
dataset=pd.read_csv('# your csv file ')
X= dataset.iloc[].values         # X is a independent variable
y= dataset.iloc[].values         # y is a dependent variable
          
#Encoding categorical data into numerical if required
        from sklearn.preprocessing import LabelEncoder,OneHotEncoder 

        labelencoder_X=LabelEncoder() 

        X[:,3]=labelencoder_X.fit_transform(X[:,3]) 

        onehotencoder=OneHotEncoder(categorical_features =[3])

        X=onehotencoder.fit_transform(X).toarray()
          
#Apply feature scaling 
  from sklearn.preprocessing import StandardScaler 
  sc_X=StandardScaler()
  sc_y=StandardScaler()
  X=sc_X.fit_transform(X)
  y=sc_y.fit_transform(y)

          
#Fitting Logistic Regression to the Training set
  from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train)

          
#Predicting the results and calculating confusion matrix
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
          
#Visualising the Training set results
          from matplotlib.colors import ListedColormap 

          X_set, y_set = X_train, y_train

          X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                                    stop = X_set[:, 0].max() + 1, step = 0.01),
                                    np.arange(start = X_set[:, 1].min() - 1,
                                    stop = X_set[:, 1].max() + 1, step = 0.01))

          plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                       X2.ravel()]).T).reshape(X1.shape),alpha = 0.75,
                      cmap = ListedColormap(('red', 'green')))

          plt.xlim(X1.min(), X1.max())

          plt.ylim(X2.min(), X2.max())

          for i, j in enumerate(np.unique(y_set)):

                    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                                c = ListedColormap(('red', 'green'))(i), label = j)

          plt.title('Logistic Regression (Training set)')

          plt.xlabel('')

          plt.ylabel('')

          plt.legend()

          plt.show()

