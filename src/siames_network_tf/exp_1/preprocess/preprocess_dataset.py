import pandas as pd
import numpy as np

import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def remove_punct(text: str) -> str:
    """
        
    """
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def clear_text(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
        
    """
    df[column_name] = df[column_name].apply(lambda x: remove_punct(str(x)))
    
    df[column_name] = df[column_name].apply(
        lambda x: [word.lower() for word in x.split() if word.lower() not in ENGLISH_STOP_WORDS]
        )
    df[column_name] = df[column_name].apply(lambda x: " ".join(x))

    return df

