# vlad_nlp_final

This repository contains a baseline for fine-tuning the Russian transformer model
`ai-forever/ruRoberta-large` on the supplied dataset and making predictions for
the test portion.

Run `train_and_predict.py` to start training and prediction. The script expects
training data in `data/train_data.csv` and evaluation data in
`data/test_data.csv` (tab separated). After training the best checkpoint is saved
inside `checkpoints/` and predictions are written to `prediction.csv` with
columns `id` and `label`.

Example:

```bash
python train_and_predict.py --epochs 3 --batch_size 8
```


## Using Conda

To install dependencies and run the script with conda:

```bash
conda create -n nlp python=3.10
conda activate nlp
pip install -r requirements.txt
python train_and_predict.py --epochs 3 --batch_size 8
```

