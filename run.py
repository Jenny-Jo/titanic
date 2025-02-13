
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

from dataset import preprocess_dataset
from model import get_model
from utils import reset_seeds

def get_cross_validation(shuffle:bool=True, is_kfold:bool=True, n_splits:int=5):
    if is_kfold:
      return KFold(n_splits=n_splits, shuffle=shuffle)
    else:
      return StratifiedKFold(n_splits=n_splits, shuffle=shuffle)

def run_cross_validation(my_model, x_train, y_train, cv, is_kfold:bool=True):
    n_iter = 0
    accuracy_lst = []
    if is_kfold:
        cross_validation = cv.split(x_train)
    else:
        cross_validation = cv.split(x_train, y_train)

    for train_index, valid_index in cross_validation:
      n_iter += 1
      # 학습용, 검증용 데이터 구성
      train_x, valid_x = x_train.iloc[train_index], x_train.iloc[valid_index]
      train_y, valid_y = y_train.iloc[train_index], y_train.iloc[valid_index]
      # 학습
      my_model.fit(train_x, train_y)
      # 예측
      pred = my_model.predict(valid_x)
      # 평가
      accuracy = np.round(accuracy_score(valid_y, pred), 4)
      accuracy_lst.append(accuracy)
      print(f'{n_iter} 번째 K-fold 정확도: {accuracy}, 학습데이터 크기: {train_x.shape}, 검증데이터 크기: {valid_x.shape}')

    return np.mean(accuracy_lst)

@reset_seeds
def main():
    # 데이터 로드 및 분류
    X_train, X_test, y_train, y_test = preprocess_dataset()
    # 모델 생성
    my_model = get_model()
    # 교차 검증
    is_Regression = False
    my_cv = get_cross_validation(is_kfold=is_Regression)
    # 모델 학습
    accuracy = run_cross_validation(my_model, X_train, y_train, my_cv, is_kfold=is_Regression)

    # 테스트 데이터 예측
    return my_model.score(X_test, y_test)

if __name__=="__main__":
  result = main()
  print(f"테스트 스코어는 {result}")
