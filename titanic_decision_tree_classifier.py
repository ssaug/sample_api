import pandas as pd
import mlflow
import mlflow.sklearn

### initilaise the mlflow tracking

## enable autologging
mlflow.sklearn.autolog()
df = pd.read_csv("titanic.csv")

### Objective
### Create a ML classifier to predict wheter a person will survive the titanic accident
df = df.fillna(0)
df.info()
df.head(3)
df["gender_enc"]=df["Sex"].astype('category').cat.codes

df["embark_enc"]=df["Embarked"].astype('category').cat.codes

X = df[["Pclass","Age","gender_enc","embark_enc","Fare","SibSp","Parch"]]
Y = df["Survived"]

### Test-Train split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

### Classifier
from sklearn.tree import DecisionTreeClassifier

# step:1 initialise the model class
model = DecisionTreeClassifier(criterion="entropy",max_depth=5)

#step:2 train the model over training data
model.fit(X_train,y_train)

#step:3 predict this over test_set
y_pred = model.predict(X_test)

### Model evaluation
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)

