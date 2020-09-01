import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('C:/Users/한상우/Downloads/owid-covid-data.csv')

df_korea = df[df['location'] == 'South Korea']

df_korea = df_korea.T[3:5]
df_korea = df_korea.T
df.reset_index().rename(columns={"index": "date"})

df_korea_date = np.array([df_korea['date']]).reshape(-1, 1)
df_korea_patient = np.array(df_korea['total_cases']).reshape(-1, 1)

days_in_future = 100

day_since = np.array([i for i in range(len(df_korea))]).reshape(-1, 1)
future_forcast = np.array([i for i in range(len(df_korea)+days_in_future)]).reshape(-1, 1)
korea_patient = np.array(df_korea_patient).reshape(-1, 1)

start = '2019-12-31'
start_date = datetime.datetime.strptime(start, '%Y-%m-%d')

future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d'))

X_train, X_test, y_train, y_test = train_test_split(day_since[:221], korea_patient[:221], test_size=0.12, shuffle=False)

svm_confirmed = SVR(shrinking=True, kernel='poly', gamma=0.01, epsilon=1, degree=3, C=0.1)
svm_confirmed.fit(X_train, y_train.ravel())
svm_pred = svm_confirmed.predict(future_forcast)


svm_test_pred = svm_confirmed.predict(X_test)

#교차 검증 해보기


#svm_confirmed = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(svm_confirmed, day_since[:221], korea_patient[:221], cv=3)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()/2))
print(y_test)
print('------------------')
print(svm_pred)
print('------------------')
#print(svm_test_pred)
plt.plot(y_test)
plt.plot(svm_pred)
plt.plot(svm_test_pred)
#plt.legend(['Test_Data', 'SVM Prediction'])

plt.show()
'''
plt.plot(df_korea_date, df_korea_patient)
plt.xlabel('Date')
plt.ylabel('Confirmed')
plt.show()



X_train = X_train.astype('int')
y_train = y_train.astype('int')
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
'''