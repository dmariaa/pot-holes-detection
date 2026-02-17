##### Split CSV to training,valid, test:

import pandas as pd
import numpy as np

# === Settings ===
# input_csv   = "balanced.csv"  # input file with columns: path,label
# train_csv   = "train.csv"
# valid_csv   = "valid.csv"
# test_csv    = "test.csv"
input_csv = "data_old/data_labels.csv"  # input file with columns: path,label
train_csv = "data_old/train_unbalanced.csv"
valid_csv = "data_old/valid_unbalanced.csv"
test_csv = "data_old/test_unbalanced.csv"
seed = 42  # for reproducible shuffling
train_frac = 0.80
valid_frac = 0.10  # test will get the remainder per-class

# === Load data ===
df = pd.read_csv(input_csv)
df["label"] = df["label"].astype(int)

# Sanity check: keep only required columns (optional)
df = df[["path", "label"]]

rng = np.random.default_rng(seed)

train_parts = []
valid_parts = []
test_parts = []

# Stratified split per label
for label, grp in df.groupby("label", sort=True):
    grp = grp.sample(frac=1.0, random_state=seed)  # shuffle within class (deterministic)
    n = len(grp)

    n_train = int(np.floor(train_frac * n))
    n_valid = int(np.floor(valid_frac * n))
    n_test = n - n_train - n_valid  # remainder to test to ensure totals match exactly

    # Slice according to counts
    train_parts.append(grp.iloc[:n_train])
    valid_parts.append(grp.iloc[n_train:n_train + n_valid])
    test_parts.append(grp.iloc[n_train + n_valid:])

# Concatenate and shuffle each split (deterministic)
train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
valid_df = pd.concat(valid_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

# Save
train_df.to_csv(train_csv, index=False)
valid_df.to_csv(valid_csv, index=False)
test_df.to_csv(test_csv, index=False)


# Print counts
def counts(name, frame):
    c = frame["label"].value_counts().sort_index()
    # Ensure every label is present (balanced split with all labels)
    missing = sorted(set(df["label"].unique()) - set(c.index))
    print(f"\n{name} counts:")
    print(c.to_string())
    if missing:
        print(f"(Missing labels in {name}: {missing})")


counts("TRAIN", train_df)
counts("VALID", valid_df)
counts("TEST", test_df)

print(
    f"\nWrote:\n- {train_csv} ({len(train_df)} rows)\n- {valid_csv} ({len(valid_df)} rows)\n- {test_csv} ({len(test_df)} rows)")
