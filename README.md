# xgboost-liberty-mutual

Tabular `xgboost` model using data from the [Liberty Mutual Group](https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction) Kaggle competition.

```
prodspell run \
  --machine-type CPU \
  --github-url https://github.com/ResidentMario/spell-xgboost-liberty-mutual.git \
  --pip xgboost --pip kaggle \
  "chmod +x /spell/scripts/download_data.sh; /spell/scripts/download_data.sh; python /spell/models/model_1.py"
```
