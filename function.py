from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder   
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
 
from sklearn.model_selection import RandomizedSearchCV
 
 
X = df.drop(columns = ["user_behavior_class"] , axis = 1)
y = df["user_behavior_class"]
 
cat_cols = X.select_dtypes(include=['object']).columns.values
num_cols = X.select_dtypes(include=np.number).columns.tolist()
 
# Data Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ])
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42, stratify=y)
 
clf = LogisticRegression(max_iter=200)
 
scores = {'accuracy': {}, 'precision': {}, 'recall': {}, 'f1-score': {}}
 
model = Pipeline(
        steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)])
 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
report = classification_report(y_test, y_pred, output_dict=True)
scores['accuracy'][name] = accuracy_score(y_test, y_pred)
scores['precision'][name] = report['macro avg']['precision']
scores['recall'][name] = report['macro avg']['recall']
scores['f1-score'][name] = report['macro avg']['f1-score']