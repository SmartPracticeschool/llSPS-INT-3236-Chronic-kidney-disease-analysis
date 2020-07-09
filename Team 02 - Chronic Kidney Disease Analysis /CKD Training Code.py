import pandas as pd

dataset = pd.read_csv("Dataset/kidney.csv")
pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)

dataset["classification"].value_counts()
dataset.isnull().any()
dataset.isnull().sum()

df = dataset.drop('id', axis=1)
df.drop([68, 187], axis=0, inplace=True)
df.drop(164, axis=0, inplace=True)

df['bp'].fillna(df['bp'].mode()[0], inplace=True)
df['sg'].fillna(df['sg'].mode()[0], inplace=True)
df['al'].fillna(df['al'].mode()[0], inplace=True)
df['su'].fillna(df['su'].mode()[0], inplace=True)
df['rbc'].fillna(df['rbc'].mode()[0], inplace=True)
df['pc'].fillna(df['pc'].mode()[0], inplace=True)
df['pcc'].fillna(df['pcc'].mode()[0], inplace=True)
df['ba'].fillna(df['ba'].mode()[0], inplace=True)
df['htn'].fillna(df['htn'].mode()[0], inplace=True)
df['dm'].fillna(df['dm'].mode()[0], inplace=True)
df['cad'].fillna(df['cad'].mode()[0], inplace=True)
df['appet'].fillna(df['appet'].mode()[0], inplace=True)
df['pe'].fillna(df['pe'].mode()[0], inplace=True)
df['ane'].fillna(df['ane'].mode()[0], inplace=True)
df['pcv'].fillna(df['pcv'].mode()[0], inplace=True)
df['wc'].fillna(df['wc'].mode()[0], inplace=True)
df['rc'].fillna(df['rc'].mode()[0], inplace=True)

df['age'].fillna(df['age'].mean(), inplace=True)
df['bgr'].fillna(df['bgr'].mean(), inplace=True)
df['bu'].fillna(df['bu'].mean(), inplace=True)
df['sc'].fillna(df['sc'].mean(), inplace=True)
df['sod'].fillna(df['sod'].mean(), inplace=True)
df['pot'].fillna(df['pot'].mean(), inplace=True)
df['hemo'].fillna(df['hemo'].mean(), inplace=True)

df.isnull().any()

x = df.iloc[:, :24].values
y = df.iloc[:, -1].values

df['classification'] = df['classification'].replace(['ckd\t'], 'ckd')

df['dm'] = df['dm'].replace(['\tno'], 'no')
df['dm'] = df['dm'].replace(['\tyes'], 'yes')
df['dm'] = df['dm'].replace([' yes'], 'yes')
df['cad'] = df['cad'].replace(['\tno'], 'no')

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
df.iloc[:, 5] = lb.fit_transform(df.iloc[:, 5])
df.iloc[:, 6] = lb.fit_transform(df.iloc[:, 6])
df.iloc[:, 7] = lb.fit_transform(df.iloc[:, 7])
df.iloc[:, 8] = lb.fit_transform(df.iloc[:, 8])

df.iloc[:, 18] = lb.fit_transform(df.iloc[:, 18])
df.iloc[:, 19] = lb.fit_transform(df.iloc[:, 19])
df.iloc[:, 20] = lb.fit_transform(df.iloc[:, 20])
df.iloc[:, 21] = lb.fit_transform(df.iloc[:, 21])
df.iloc[:, 22] = lb.fit_transform(df.iloc[:, 22])
df.iloc[:, 23] = lb.fit_transform(df.iloc[:, 23])
df.iloc[:, 24] = lb.fit_transform(df.iloc[:, 24])
dataset["classification"].value_counts()

q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
IQR = q3 - q1
IQR

df = df[~((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR))).any(axis=1)]
df["classification"].value_counts()
x = df.iloc[:, 0:24].values
y = df.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(x)

x.tolist()

from joblib import dump

dump(sc, "Model Files/xtransform.save")

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

from sklearn.utils import resample
from sklearn.svm import SVC

scores = []
best_svc = SVC(kernel='rbf', gamma='auto', degree=3, random_state=0)
cv = KFold(n_splits=100, random_state=48, shuffle=True)
for train_index, test_index in cv.split(x):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
    best_svc.fit(X_train, y_train)
    scores.append(best_svc.score(X_test, y_test))

dump(best_svc, "Model Files/kidneysvm.save")