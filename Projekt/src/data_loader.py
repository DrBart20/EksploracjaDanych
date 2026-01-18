import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

def load_california_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
 
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Target'] = housing.target

    X = df.drop('Target', axis=1)
    y = df['Target']
    
    return df, X, y

def load_csv_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if df.shape[1] < 2:
                return None, None, None
                
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            return df, X, y
        
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku CSV: {e}")
            return None, None, None
    
    return None, None, None

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> Tuple:

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test
