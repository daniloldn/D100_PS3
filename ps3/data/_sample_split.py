import hashlib

import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """

    # 1. Get unique IDs
    ids = df[id_column].unique()

    # 2. Hash each unique ID
    id_hashes = {
        _id: int(hashlib.sha256(str(_id).encode()).hexdigest(), 16)
        for _id in ids
    }

    # 3. Sort IDs by hash value
    sorted_ids = sorted(id_hashes, key=lambda k: id_hashes[k])

    # 4. Select the top X% of IDs for training
    n_train = int(training_frac * len(sorted_ids))
    train_ids = set(sorted_ids[:n_train])

    # 5. Assign each row to train (1) or test (0)
    df["sample"] = df[id_column].apply(lambda x: 1 if x in train_ids else 0)

    return df
