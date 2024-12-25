import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer

SEED = 11

def load_data(path:str):
    """take the path that conntain the data

    Args:
        path (str): the file that contain the data NOTE:Do not pass the data's path it self

    Returns:
        DataFrame: the data splited to train and val 
    """
    X_train = pd.read_csv(path+"train.csv")
    y_train = X_train["Class"]
    
    X_val = pd.read_csv(path+"val.csv")    
    y_val = X_val['Class']
    
    X_train.drop(columns="Class", inplace=True)
    X_val.drop(columns="Class", inplace=True)
    return X_train, y_train, X_val, y_val

def scale_data(X_train:pd.DataFrame, scaler_type="standard"):
    """fitting the data to one of the avalipale scaler (MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer) 

    Args:
        X_train (pd.DataFrame): the train data
        scaler_type (str, optional): the sclaer. Defaults to "standard".

    Returns:
        pd.DataFrame: the fitted data
        Scaler: the used scaler
    """
    if(scaler_type == "standard"):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        return X_train, scaler
    elif(scaler_type == "minmax"):
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        return X_train, scaler
    elif(scaler_type == "robus"):
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        return X_train, scaler
    elif(scaler_type == "power"):
        scaler = PowerTransformer()
        X_train = scaler.fit_transform(X_train)
        return X_train, scaler
    
    else:
        raise ValueError(f"Invalid technique: {scaler_type}. Choose from ['standard', 'minmax', 'robus', 'power']")
               
def do_the_scale(X_train:pd.DataFrame, x_val:pd.DataFrame, sclaer = "standard"):
    """perform the scaling on the data

    Args:
        X_train (pd.DataFrame): train dataset
        x_vald (pd.DataFrame): valdion dataset
        sclaer (str, optional): the scaler to perform on the dataa. Defaults to "standard".

    Returns:
        DataFrame: the scaled DataFrame
    """
    X_train, scaler = scale_data(X_train, sclaer)
    
    x_val = scaler.transform(x_val)
    
    return X_train, x_val

def solve_imbalance(x ,y ,technique:str):
    """Handling the imbalance in the data

    Args:
        x (DataFrame): the data you want to resampled
        y (DataFrame): the target of the data
        technique (str): the technique the will be chossed you can Choose from ['rus', 'nearmiss', 'smote', 'adasyn', 'smoteenn', 'smotetomek'] 

    Raises:
        ValueError:

    Returns:
        DataFrame: reasampled data
    """
    if(technique == "rus"):
        rus = RandomUnderSampler(sampling_strategy='majority', random_state=SEED)
        x_rus, y_rus = rus.fit_resample(x, y)
        return x_rus, y_rus
    elif(technique == 'nearmiss'):
        rus = NearMiss(sampling_strategy='majority', random_state=SEED)
        x_rus, y_rus = rus.fit_resample(x, y)
        return x_rus, y_rus
    ############# OverSampling ############
    elif(technique == "smote"):
        rus = SMOTE(random_state=SEED)
        x_rus, y_rus = rus.fit_resample(x, y)
        return x_rus, y_rus
    elif(technique == "adasyn"):
        rus = ADASYN(random_state=SEED)
        x_rus, y_rus = rus.fit_resample(x, y)
        return x_rus, y_rus
    ############# Mix Over and Under Sampling #############
    elif(technique == "smoteenn"):
        rus = SMOTEENN(random_state=SEED)
        x_rus, y_rus = rus.fit_resample(x, y)
        return x_rus, y_rus
    elif(technique == "smotetomek"):
        rus = SMOTETomek(random_state=SEED)
        x_rus, y_rus = rus.fit_resample(x, y)
        return x_rus, y_rus
    else:
        raise ValueError(f"Invalid technique: {technique}. Choose from ['rus', 'nearmiss', 'smote', 'adasyn', 'smoteenn', 'smotetomek']")

