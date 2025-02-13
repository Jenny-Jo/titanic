
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from utils import reset_seeds
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

# 타이타닉 데이터 로드
def __load_data() -> pd.DataFrame:
    return pd.read_csv("./data/train.csv")

def __process_drop(train, test):
    drop_cols = ['name', 'ticket', 'cabin']

    train.drop(drop_cols, axis=1, inplace=True) # 모델이 학습하는데 사용하는 데이터
    test.drop(drop_cols, axis=1, inplace=True) # 모델의 학습을 평가(잘했는지?? 못했는지??)하기 위한 데이터

def __process_null(train, test):
    age_median = train['age'].median()
    fare_median = train['fare'].median()
    embarked_mode = train['embarked'].mode().values[0]

    train['age'].fillna(age_median, inplace=True)
    test['age'].fillna(age_median, inplace=True)

    train['fare'].fillna(fare_median, inplace=True)
    test['fare'].fillna(fare_median, inplace=True)

    train['embarked'].fillna(embarked_mode, inplace=True)
    test['embarked'].fillna(embarked_mode, inplace=True)

def __preprocess_resample(train, test):
    print("__preprocess_resample start")
    print(f"train.shape: {train.shape} / test.shape: {test.shape}")
    X_train, y_train = SMOTE().fit_resample(train.drop(['survived'], axis=1), train['survived'])
    X_test, y_test = SMOTE().fit_resample(test.drop(['survived'], axis=1), test['survived'])

    print("__preprocess_resample end")
    print(f"X_train.shape: {X_train.shape} / X_test.shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def __preprocess_encoding(train, test):
    results = []

    cat_features = ['gender','embarked']
    train[cat_features] = train[cat_features].astype('category')
    test[cat_features] = test[cat_features].astype('category')
    normal_cols = list(set(train.columns) - set(cat_features))

    enc = OneHotEncoder()
    enc.fit(train[cat_features])

    pd_list = [train, test]
    for i, df in enumerate(pd_list, start=1):
      _df = pd.DataFrame(
        enc.transform(df[cat_features]).toarray(),
        columns = enc.get_feature_names_out()
      )
      results.append(
        pd.concat(
          [df[normal_cols].reset_index(drop=True), _df.reset_index(drop=True)]
          , axis=1
        ).reset_index(drop=True)
      )
    return results[0], results[1]


def __preprocess_data(train, test):
    print(f'before: {train.shape} / {test.shape}')
    # 필요없는 컬럼 제거
    __process_drop(train, test)
    # 결측치 처리
    __process_null(train, test)
    # 범주형 처리
    return __preprocess_encoding(train, test)

@reset_seeds
def preprocess_dataset():
    # 데이터 로드
    df_raw = __load_data()
    # 데이터 분리
    train, test = train_test_split(df_raw, test_size=0.2, stratify=df_raw['survived'])
    # 데이터 전처리
    train, test = __preprocess_data(train, test)
    # features, target 분리
    return __preprocess_resample(train, test)
    # return train.drop(['survived'], axis=1), test.drop(['survived'], axis=1), train['survived'], test['survived']
