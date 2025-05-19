- name: baseline-bert
  entry: train_and_predict.py
  description: Fine-tune ruBERT and predict on test

## baseline-bert

### Краткая цель
Скрипт `train_and_predict.py` **дообучает** ai-forever/ruRoberta-large на `data/train.json`  
и **делает инференс** на `data/test.json`, сохраняя `prediction.csv`  
(колонки `id,label`) и лучший чекпойнт в `checkpoints/`.

для шедулинга можно попробовать
https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
Installing:
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
Example:
>> from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
>>
>> model = ...
>> optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5) # lr is min lr
>> scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=200,
                                          cycle_mult=1.0,
                                          max_lr=0.1,
                                          min_lr=0.001,
                                          warmup_steps=50,
                                          gamma=1.0)
>> for epoch in range(n_epoch):
>>     train()
>>     valid()
>>     scheduler.step()

### Зависимости
```txt
transformers>=4.40
datasets
scikit-learn
torch>=2.0
tqdm
