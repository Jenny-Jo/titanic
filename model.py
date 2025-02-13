
from lightgbm import LGBMClassifier, plot_importance
from utils import reset_seeds

# 모델 생성 후 리턴
@reset_seeds
def get_model(hp:dict=None, model_nm:str=None):
    if not hp:
        hp = {"verbose":-1} # warning 로그 제거

    if not model_nm:
        return LGBMClassifier(**hp)
    elif model_nm == "LGBMClassifier":
        return LGBMClassifier(**hp)
