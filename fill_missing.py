import pandas as pd
import numpy as np

def fill_null_with_most_common(df, key_column, value_column, inplace=False):
    key_unique = df[key_column].unique()
    
    most_common_value = {
        key: df[df[key_column] == key][value_column].mode(dropna=True)
        for key in key_unique
    }
    
    def fill(row):
        if pd.isna(row[value_column]):
            try:
                return most_common_value[row[key_column]].values[0]
            except:
                return np.nan
        else:
            return row[value_column]
    
    if inplace:
        df[value_column] = df.apply(fill, axis=1)
    else:
        df = df.copy()
        df[value_column] = df.apply(fill, axis=1)
        return df
