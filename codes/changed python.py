import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df_ROS = pd.read_csv('datasets/RecurrenceOfSurgery.csv')
print(df_ROS)

df_ROS[['통증기간(월)', '헤모글로빈수치','스테로이드치료', '연령', '흡연여부', '입원기간', '고혈압여부']].isnull().sum()

# df_ROS에서 누락된 값을 예측하려는 열과 다른 열을 추출
columns_to_predict = ['통증기간(월)', '헤모글로빈수치']  # 예측하려는 열
other_columns = ['스테로이드치료', '연령', '흡연여부', '입원기간', '고혈압여부']  # 다른 열

# 누락된 값을 예측하기 위한 학습 데이터와 테스트 데이터로 나누기
X_train = df_ROS[other_columns][df_ROS['통증기간(월)'].notnull()]
y_train = df_ROS['통증기간(월)'][df_ROS['통증기간(월)'].notnull()]
X_test = df_ROS[other_columns][df_ROS['통증기간(월)'].isnull()]


# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(X_train, y_train)


# 누락된 값을 예측하여 채우기
predicted_missing_values = model.predict(X_test)

# 예측된 값을 데이터프레임에 채워 넣기
df_ROS.loc[df_ROS['통증기간(월)'].isnull(), '통증기간(월)'] = predicted_missing_values


df_ROS[['통증기간(월)', '헤모글로빈수치','스테로이드치료', '연령', '흡연여부', '입원기간', '고혈압여부']].isnull().sum()

# 누락된 값을 예측하기 위한 학습 데이터와 테스트 데이터로 나누기
X_train = df_ROS[other_columns][df_ROS['헤모글로빈수치'].notnull()]
y_train = df_ROS['헤모글로빈수치'][df_ROS['헤모글로빈수치'].notnull()]
X_test = df_ROS[other_columns][df_ROS['헤모글로빈수치'].isnull()]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(X_train, y_train)


# 누락된 값을 예측하여 채우기
predicted_missing_values = model.predict(X_test)

# 예측된 값을 데이터프레임에 채워 넣기
df_ROS.loc[df_ROS['헤모글로빈수치'].isnull(), '헤모글로빈수치'] = predicted_missing_values


df_ROS[['통증기간(월)', '헤모글로빈수치','스테로이드치료', '연령', '흡연여부', '입원기간', '고혈압여부']].isnull().sum()

df_ROS_ex = df_ROS[['통증기간(월)', '헤모글로빈수치','스테로이드치료', '연령', '흡연여부', '입원기간', '고혈압여부']]
print(df_ROS_ex)

# 정형화
target = df_ROS_ex['스테로이드치료']
features = df_ROS_ex[['연령', '통증기간(월)', '헤모글로빈수치', '입원기간' ]]
print(target.shape, features.shape)

# Split
from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=111)
print(features_train.shape, target_train.shape, features_test.shape, target_test.shape)

# 모델학습
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

from sklearn.model_selection import GridSearchCV

hyper_params = {'min_samples_leaf' : [5, 7, 9]
               ,'max_depth' : [9, 11]
               ,'min_samples_split' : [5, 6, 7]}

# 평가 score 분류
from sklearn.metrics import f1_score, make_scorer
scoring = make_scorer(f1_score)

grid_search = GridSearchCV(model, param_grid=hyper_params, cv=3, verbose=1, scoring=scoring) 

grid_search.fit(features_train, target_train)

grid_search.best_estimator_

grid_search.best_score_, grid_search.best_params_ 

best_model = grid_search.best_estimator_ # 하나의 모델 --> 그 중에서 최고의 모델
print(best_model)  

target_test_predict = best_model.predict(features_test)
print(target_test_predict)

# 평가
from sklearn.metrics import classification_report

print(classification_report(target_test, target_test_predict))