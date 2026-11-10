# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## How to run the pipeline

```bash
python -m pip install -r requirements.txt
python train.py          # cleans, validates, fits, writes validation_predictions.csv
                         # and fills data/december_chart_inputs.csv
python test_pipeline.py  # 5 assert-based checks on cleaning/features/prediction
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`train.py` prints the hyperparameter sweep and the full validation metrics table
(also in `report.md`). Everything is deterministic (fixed random seed).

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
