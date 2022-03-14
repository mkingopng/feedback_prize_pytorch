import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

df = pd.read_csv("../data/train.csv")
dfx = pd.get_dummies(df, columns=["discourse_type"]).groupby(["id"], as_index=False).sum()
cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
dfx = dfx[cols]
mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=42)
labels = [c for c in dfx.columns if c != "id"]
dfx_labels = dfx[labels]
dfx["kfold"] = -1
for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
    print(len(trn_), len(val_))
    dfx.loc[val_, "kfold"] = fold
df = df.merge(dfx[["id", "kfold"]], on="id", how="left")
print(df.kfold.value_counts())
df.to_csv("2_train_folds.csv", index=False)

