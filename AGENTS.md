- name: baseline-bert
  entry: train_and_predict.py
  description: fix baseline for following train and test structure
## baseline-bert

### цель
Нужно исправить train_and_predict.py чтобы он правильно работал со структурой данных

формат данных:
sentence - предложение
entity - объект анализа тональности
entity_tag - тип сущности в рамках PERSON, ORGANIZATION, PROFESSION, COUNTRY, NATIONALITY
entity_pos_start_rel - start pos of entity in sentence
entity_pos_end_rel - end pos of entity in sentence
то есть, entity = sentence[entity_pos_start_rel:entity_pos_end_rel]
label - метка тональности (0 - нейтрально, -1 - отрицательно, 1 - положительно)
