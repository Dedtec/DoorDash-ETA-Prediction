import pandas as pd
import numpy as np
import pytest
from fill_missing import fill_null_with_most_common

def test_fill_null_with_most_common():
    data = {
        'store_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'store_primary_category': ['A', 'A', np.nan, 'B', np.nan, 'C', 'C', 'D', np.nan]
    }
    df = pd.DataFrame(data)

    expected_data = {
        'store_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'store_primary_category': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'C']
    }
    expected_df = pd.DataFrame(expected_data)

    # Test with inplace=False (default)
    filled_df = fill_null_with_most_common(df.copy(), 'store_id', 'store_primary_category')
    pd.testing.assert_frame_equal(filled_df, expected_df)

    # Test with inplace=True
    df_copy = df.copy()
    fill_null_with_most_common(df_copy, 'store_id', 'store_primary_category', inplace=True)
    pd.testing.assert_frame_equal(df_copy, expected_df)

def test_null_only_keys():
    data = {
        'store_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
        'store_primary_category': ['A', 'A', np.nan, 'B', np.nan, 'C', 'C', 'D', np.nan, np.nan, np.nan, np.nan]
    }
    df = pd.DataFrame(data)

    expected_data = {
        'store_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
        'store_primary_category': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'C', np.nan, np.nan, np.nan]
    }
    expected_df = pd.DataFrame(expected_data)

    fill_null_with_most_common(df, 'store_id', 'store_primary_category', inplace=True)
    pd.testing.assert_frame_equal(df, expected_df)

def test_original_intact_with_inplace_false():
    data = {
        'store_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'store_primary_category': ['A', 'A', np.nan, 'B', np.nan, 'C', 'C', 'D', np.nan]
    }
    df = pd.DataFrame(data)
    df_original = df.copy()  # Make a copy of the original DataFrame

    # Test with inplace=False
    fill_null_with_most_common(df, 'store_id', 'store_primary_category', inplace=False)

    # Check that the original DataFrame is intact
    pd.testing.assert_frame_equal(df, df_original)

if __name__ == "__main__":
    pytest.main()
