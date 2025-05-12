from google.colab import files
uploaded = files.upload()
import pandas as pd
df = pd.read_excel('NAFLD.xlsx',engine ='openpyxl')

#Random Forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
df = pd.read_excel('NAFLD.xlsx', engine='openpyxl')
X = df.drop(columns=['Diagnosis according to SAF (NASH=1, NAFL=2)',
                     'Type of Disease (Mild illness=1, Severe illness=2)'])
y = df[['Diagnosis according to SAF (NASH=1, NAFL=2)',
        'Type of Disease (Mild illness=1, Severe illness=2)']]
X = X.fillna(X.mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_combined = y_train['Diagnosis according to SAF (NASH=1, NAFL=2)'].astype(str) + "_" + \
                   y_train['Type of Disease (Mild illness=1, Severe illness=2)'].astype(str)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_combined_resampled = smote.fit_resample(X_train, y_train_combined)
y_train_resampled = pd.DataFrame(
    [label.split("_") for label in y_train_combined_resampled],
    columns=['Diagnosis according to SAF (NASH=1, NAFL=2)', 'Type of Disease (Mild illness=1, Severe illness=2)']
).astype(int)
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)
rf_diagnosis = RandomForestClassifier(
    n_estimators=20,
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight={1: 0.5, 2: 0.5},
    random_state=42
)
rf_severity = RandomForestClassifier(
    n_estimators=20,
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight={1: 0.7, 2: 0.3},
    random_state=42
)
rf_diagnosis.fit(X_train_resampled, y_train_resampled['Diagnosis according to SAF (NASH=1, NAFL=2)'])
rf_severity.fit(X_train_resampled, y_train_resampled['Type of Disease (Mild illness=1, Severe illness=2)'])
y_preds_diagnosis = rf_diagnosis.predict(X_test)
y_preds_severity = rf_severity.predict(X_test)
print("Diagnosis Classification Report:")
print(classification_report(y_test['Diagnosis according to SAF (NASH=1, NAFL=2)'], y_preds_diagnosis))
print("\nSeverity Classification Report:")
print(classification_report(y_test['Type of Disease (Mild illness=1, Severe illness=2)'], y_preds_severity))
cm_diagnosis = confusion_matrix(y_test['Diagnosis according to SAF (NASH=1, NAFL=2)'], y_preds_diagnosis)
ConfusionMatrixDisplay(confusion_matrix=cm_diagnosis, display_labels=['NAFL', 'NASH']).plot(cmap='Blues')
plt.title('Confusion Matrix: Diagnosis')
plt.show()
cm_severity = confusion_matrix(y_test['Type of Disease (Mild illness=1, Severe illness=2)'], y_preds_severity)
ConfusionMatrixDisplay(confusion_matrix=cm_severity, display_labels=['Mild', 'Severe']).plot(cmap='Greens')
plt.title('Confusion Matrix: Severity')
plt.show()

#XGBoost
from sklearn.model_selection import  train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer,accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
df = pd.read_excel('NAFLD.xlsx', engine='openpyxl')
x = df.drop(columns=['Diagnosis according to SAF (NASH=1, NAFL=2)',
                     'Type of Disease (Mild illness=1, Severe illness=2)'])
y = df[['Diagnosis according to SAF (NASH=1, NAFL=2)',
        'Type of Disease (Mild illness=1, Severe illness=2)']]
x = x.fillna(x.mean())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_train_combined = y_train['Diagnosis according to SAF (NASH=1, NAFL=2)'].astype(str) + "_" + \
                   y_train['Type of Disease (Mild illness=1, Severe illness=2)'].astype(str)
smote = SMOTE(random_state=42)
x_train_resampled, y_train_combined_resampled = smote.fit_resample(x_train, y_train_combined)
y_train_resampled = pd.DataFrame(
    [label.split("_") for label in y_train_combined_resampled],
    columns=['Diagnosis according to SAF (NASH=1, NAFL=2)', 'Type of Disease (Mild illness=1, Severe illness=2)']
).astype(int)
scaler = StandardScaler()
x_train_resampled = scaler.fit_transform(x_train_resampled)
x_test = scaler.transform(x_test)
np.random.seed(42)
noise = np.random.normal(0, 0.5, size=x_train_resampled.shape)
x_train_resampled_noisy = x_train_resampled + noise
xgb_model = XGBClassifier(
    gamma=5,
    alpha=1,
    reg_lambda=0.1,
    n_estimators=100,
    max_depth=2,
    random_state=42,
    eval_metric='mlogloss'
)
multi_target_model = MultiOutputClassifier(xgb_model)
y_train_resampled['Diagnosis according to SAF (NASH=1, NAFL=2)'] -= 1
y_train_resampled['Type of Disease (Mild illness=1, Severe illness=2)'] -= 1
y_test['Diagnosis according to SAF (NASH=1, NAFL=2)'] -= 1
y_test['Type of Disease (Mild illness=1, Severe illness=2)'] -= 1
multi_target_model.fit(x_train_resampled_noisy, y_train_resampled)
y_preds = multi_target_model.predict(x_test)
print("\nDiagnosis Classification Report:")
print(classification_report(y_test['Diagnosis according to SAF (NASH=1, NAFL=2)'], y_preds[:, 0]))
print("\nSeverity Classification Report:")
print(classification_report(y_test['Type of Disease (Mild illness=1, Severe illness=2)'], y_preds[:, 1]))
cm_diagnosis = confusion_matrix(y_test['Diagnosis according to SAF (NASH=1, NAFL=2)'], y_preds[:, 0])
ConfusionMatrixDisplay(confusion_matrix=cm_diagnosis, display_labels=['NAFL', 'NASH']).plot(cmap='Blues')
plt.title('XGBoost Confusion Matrix: Diagnosis')
plt.show()
cm_severity = confusion_matrix(y_test['Type of Disease (Mild illness=1, Severe illness=2)'], y_preds[:, 1])
ConfusionMatrixDisplay(confusion_matrix=cm_severity, display_labels=['Mild', 'Severe']).plot(cmap='Greens')
plt.title('XGBoost Confusion Matrix: Severity')
plt.show()

#Logistic Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
file_name = "NAFLD.xlsx"
data = pd.read_excel(file_name, engine='openpyxl')
x = data.drop(columns=['Diagnosis according to SAF (NASH=1, NAFL=2)', 'Type of Disease (Mild illness=1, Severe illness=2)'])
y = data[['Diagnosis according to SAF (NASH=1, NAFL=2)', 'Type of Disease (Mild illness=1, Severe illness=2)']]
x = x.fillna(x.mean(numeric_only=True))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_train_combined = y_train['Diagnosis according to SAF (NASH=1, NAFL=2)'].astype(str) + "_" + \
                   y_train['Type of Disease (Mild illness=1, Severe illness=2)'].astype(str)
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled_combined = smote.fit_resample(x_train, y_train_combined)
y_train_resampled = pd.DataFrame(
    [row.split("_") for row in y_train_resampled_combined],
    columns=['Diagnosis according to SAF (NASH=1, NAFL=2)', 'Type of Disease (Mild illness=1, Severe illness=2)']
).astype(int)
scaler = StandardScaler()
x_train_resampled = scaler.fit_transform(x_train_resampled)
x_test = scaler.transform(x_test)
logistic_model = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
multi_target_model = MultiOutputClassifier(logistic_model)
multi_target_model.fit(x_train_resampled, y_train_resampled)
y_preds = multi_target_model.predict(x_test)
print("\nDiagnosis Classification Report:")
print(classification_report(y_test['Diagnosis according to SAF (NASH=1, NAFL=2)'], y_preds[:, 0]))
print("\nSeverity Classification Report:")
print(classification_report(y_test['Type of Disease (Mild illness=1, Severe illness=2)'], y_preds[:, 1]))
cm_diagnosis = confusion_matrix(y_test['Diagnosis according to SAF (NASH=1, NAFL=2)'], y_preds[:, 0])
ConfusionMatrixDisplay(confusion_matrix=cm_diagnosis, display_labels=['NAFL', 'NASH']).plot(cmap='Blues')
plt.title('Logistic regression Confusion Matrix: Diagnosis')
plt.show()
cm_severity = confusion_matrix(y_test['Type of Disease (Mild illness=1, Severe illness=2)'], y_preds[:, 1])
ConfusionMatrixDisplay(confusion_matrix=cm_severity, display_labels=['Mild', 'Severe']).plot(cmap='Greens')
plt.title('Logistic regression Confusion Matrix: Severity')
plt.show()


#--#Support Vector Machine-SVM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
df = pd.read_excel('NAFLD.xlsx', engine='openpyxl')
x = df.drop(columns=['Diagnosis according to SAF (NASH=1, NAFL=2)',
                     'Type of Disease (Mild illness=1, Severe illness=2)'])
y = df[['Diagnosis according to SAF (NASH=1, NAFL=2)',
        'Type of Disease (Mild illness=1, Severe illness=2)']]
x = x.fillna(x.mean(numeric_only=True))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_train_combined = y_train['Diagnosis according to SAF (NASH=1, NAFL=2)'].astype(str) + "_" + \
                   y_train['Type of Disease (Mild illness=1, Severe illness=2)'].astype(str)
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled_combined = smote.fit_resample(x_train, y_train_combined)
y_train_resampled = pd.DataFrame(
    [row.split("_") for row in y_train_resampled_combined],
    columns=['Diagnosis according to SAF (NASH=1, NAFL=2)', 'Type of Disease (Mild illness=1, Severe illness=2)']
).astype(int)
scaler = StandardScaler()
x_train_resampled = scaler.fit_transform(x_train_resampled)
x_test = scaler.transform(x_test)
svm_model = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42, probability=True)
multi_target_model = MultiOutputClassifier(svm_model)
multi_target_model.fit(x_train_resampled, y_train_resampled)
y_preds = multi_target_model.predict(x_test)
print("\nDiagnosis Classification Report:")
print(classification_report(y_test['Diagnosis according to SAF (NASH=1, NAFL=2)'], y_preds[:, 0]))
print("\nSeverity Classification Report:")
print(classification_report(y_test['Type of Disease (Mild illness=1, Severe illness=2)'], y_preds[:, 1]))
cm_diagnosis = confusion_matrix(y_test['Diagnosis according to SAF (NASH=1, NAFL=2)'], y_preds[:, 0])
ConfusionMatrixDisplay(confusion_matrix=cm_diagnosis, display_labels=['NAFL', 'NASH']).plot(cmap='Blues')
plt.title('Confusion Matrix: Diagnosis')
plt.show()
cm_severity = confusion_matrix(y_test['Type of Disease (Mild illness=1, Severe illness=2)'], y_preds[:, 1])
ConfusionMatrixDisplay(confusion_matrix=cm_severity, display_labels=['Mild', 'Severe']).plot(cmap='Greens')
plt.title('Confusion Matrix: Severity')
plt.show()

##KNN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
df = pd.read_excel('NAFLD.xlsx', engine='openpyxl')
x = df.drop(columns=['Diagnosis according to SAF (NASH=1, NAFL=2)',
                     'Type of Disease (Mild illness=1, Severe illness=2)'])
y = df[['Diagnosis according to SAF (NASH=1, NAFL=2)',
        'Type of Disease (Mild illness=1, Severe illness=2)']]
x = x.fillna(x.mean(numeric_only=True))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_train_combined = y_train['Diagnosis according to SAF (NASH=1, NAFL=2)'].astype(str) + "_" + \
                   y_train['Type of Disease (Mild illness=1, Severe illness=2)'].astype(str)
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled_combined = smote.fit_resample(x_train, y_train_combined)
y_train_resampled = pd.DataFrame(
    [row.split("_") for row in y_train_resampled_combined],
    columns=['Diagnosis according to SAF (NASH=1, NAFL=2)', 'Type of Disease (Mild illness=1, Severe illness=2)']
).astype(int)
scaler = StandardScaler()
x_train_resampled = scaler.fit_transform(x_train_resampled)
x_test = scaler.transform(x_test)
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
multi_target_model = MultiOutputClassifier(knn_model)
multi_target_model.fit(x_train_resampled, y_train_resampled)
y_preds = multi_target_model.predict(x_test)
print("\nDiagnosis Classification Report:")
print(classification_report(y_test['Diagnosis according to SAF (NASH=1, NAFL=2)'], y_preds[:, 0]))
print("\nSeverity Classification Report:")
print(classification_report(y_test['Type of Disease (Mild illness=1, Severe illness=2)'], y_preds[:, 1]))
cm_diagnosis = confusion_matrix(y_test['Diagnosis according to SAF (NASH=1, NAFL=2)'], y_preds[:, 0])
ConfusionMatrixDisplay(confusion_matrix=cm_diagnosis, display_labels=['NAFL', 'NASH']).plot(cmap='Blues')
plt.title('Confusion Matrix: Diagnosis')
plt.show()
cm_severity = confusion_matrix(y_test['Type of Disease (Mild illness=1, Severe illness=2)'], y_preds[:, 1])
ConfusionMatrixDisplay(confusion_matrix=cm_severity, display_labels=['Mild', 'Severe']).plot(cmap='Greens')
plt.title('Confusion Matrix: Severity')
plt.show()














