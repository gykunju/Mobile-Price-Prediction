from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd


train_data = pd.read_csv('./train.csv')

X = train_data.drop(columns=['price_range'])
y = train_data['price_range']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

estimators = [
    ('svc', SVC(degree=1, kernel='linear', C=0.7)),  
    ('log_reg', LogisticRegression(penalty=None, max_iter=1000)),    
]

stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(penalty=None, max_iter=1000)).fit(X_train, y_train)



ytrain_stack_pred = stack_model.predict(X_train)
ytest_stack_pred = stack_model.predict(X_test)

