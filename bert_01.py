# Setup


## pip kits


# 恢復原廠設定後查看TensorFlow的版本再指定相容版本。
# *   tensorflow-text：對文字資料進行處理。
# *   transformers：在NLU和NLG任務上具有高效能且降低預訓練計算成本

# !pip install -q tensorflow-text==2.7.0
# !pip install -q tf-models-official==2.7.0
# !pip install simpletransformers
# !pip install -U transformers


## kits version

# !pip show tensorflow
# !pip show tensorflow-text
# !pip show tf-models-official

# print("-" * 50)
# !pip show numpy
# !pip show transformers
# !pip show simpletransformers

## import kits

import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

tf.get_logger().setLevel('ERROR') # logger 記錄器

# Data

## unzip data

# !unzip "/content/drive/MyDrive/test_data/test_data_bert_01.zip" -d "/content/drive/MyDrive/test_data"

## split data

#切分訓練、驗證及測試集

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    ###"/content/drive/MyDrive/test_data/test_data_bert_01",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=seed)

class_names = raw_train_ds.class_names
print("class name:\t", class_names)

# 驗證集validation set
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "/content/drive/MyDrive/test_data/test_data_bert_01",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

## check data

for text_batch, label_batch in train_ds.take(1):
  print(text_batch)
  for i in range(3):
    print(f'Review: {text_batch.numpy()[i]}')
    label = label_batch.numpy()[i]
    print(f'Label : {label} ({class_names[label]})')
    print(label, type(label))

# Preprocessing model
'''
官方模型來源：
https://tfhub.dev/s?language=zh-tw&subtype=module,placeholder&tf-version=tf2

Bert有7個特徵屬性。現在要做的任務是文字分類，其中四個特徵是不需要的。
*   input_ids: 輸入的token對應的id
*   input_mask: 輸入的mask，1代表是正常輸入，0代表的是padding的輸入
*   segment_ids: 輸入的0：代表句子A或者padding句子，1代表句子B
*   masked_lm_positions：我們mask的token的位置
*   masked_lm_ids：我們mask的token的對應id
*   masked_lm_weights：我們mask的token的權重，1代表是真實mask的，0代表的是padding的mask
*   next_sentence_labels：句子A和B是否是上下句

'''

# 載入前處理層及bert模型
tfhub_handle_preprocess = "https://hub.tensorflow.google.cn/tensorflow/bert_zh_preprocess/3"
tfhub_handle_encoder = "https://hub.tensorflow.google.cn/tensorflow/bert_zh_L-12_H-768_A-12/4"

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)

# 定義中文模型(multi-cased多語言)
# bert_model_name = "small_bert/distilbert_multi_cased_L-6_H-768_A-12"
bert_model_name = "small_bert/bert_zh_L-12_H-768_A-12"

# 中文前處理測試
text_test = ["我來自長庚大學"]
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')

# 前處理有三個輸出層(Keys對應)
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

# 檢視模型輸出格式（為訓練其數值無意義）
bert_results = bert_model(text_preprocessed)

print(f'1.Loaded BERT: {tfhub_handle_encoder}')

# 池化輸出值：pooled_output：維度(batch_size, hidden_size)，每個sequence第一個位置[CLS]的向量輸出，用於分類任務。
print(f'2.Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')

# 序列輸出值sequence_output：維度(batch_size, seq_length, hidden_size)，這是訓練後每個token的詞向量[SEP]。
print(f'3.Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'4.Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

# Define model

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text') # 定義輸入層
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing') # 預處理層
  encoder_inputs = preprocessing_layer(text_input) # encoder_inputs編碼輸入層
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net) # dropout 隨機刪除節點防止過渡擬合
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()
epochs = 5 # 迭代次數
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy() # 實驗基數
num_train_steps = steps_per_epoch * epochs
print(num_train_steps)
num_warmup_steps = int(0.1*num_train_steps)

# Training

##chose loss function

# 二分法:loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 多分類（multi_class）
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# 指標
metrics = tf.metrics.BinaryAccuracy()

## set optimizer

epochs = 5 # 迭代次數
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy() # 實驗基數
num_train_steps = steps_per_epoch * epochs
print(num_train_steps)
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(
    init_lr=init_lr,
    num_train_steps=num_train_steps,                                          num_warmup_steps=num_warmup_steps,
    optimizer_type='adamw')

## loading and training

classifier_model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

print(f'Training model with {tfhub_handle_encoder}')

history = classifier_model.fit(
    x=train_ds,
    validation_data=val_ds,
    epochs=epochs)

# evaluate

# loss, accuracy = classifier_model.evaluate()

# print(f'1.Loss: {loss}')
# print(f'2.Accuracy: {accuracy}')

## Plot the accuracy and loss over time

history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# r is for "solid red line"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

## Export for inference

dataset_name = 'imdb'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

classifier_model.save(saved_model_path, include_optimizer=False)

reloaded_model = tf.saved_model.load(saved_model_path)

import numpy as np

def print_my_examples(inputs, results):
  result_for_printing = \
    [f"input: {inputs[i]:<30} : class: {class_names[np.argmax(results[i])] }"
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()

examples1 = ["很堅強，但這些堅強就像紙片一樣堆疊起來，輕輕一碰就倒了。 事實證明，只要有重大的變故，那些堅強，真的不夠用。 但願我有好幾顆心臟能承受，這些憂鬱又黑暗的心情。 明天，我還得要，用那制式的笑容面對大家，要假裝這一切沒發生過。"]
example_result1 = classifier_model(tf.constant(examples1))
examples2 = ["今天，我又自殺了。 額頭上出現一顆一顆紅點，是因為用腰帶勒緊脖子，導致腦缺氧。 這也不是第一次自殺了，從罹患憂鬱症以後就不斷地想死，也試過各種方法，也寫過遺書。 覺得頭快炸開了，覺得難以呼吸，痛苦地呻吟著。 第一次差點成功，一邊掉眼淚一邊自殺。 為什麼到最後作罷了？我也不知道。 但真的覺得離死亡好近，這也是我一直渴求的。 這種憂鬱到想死的心情，恐怕身邊的人都不明白吧。 因為心被掏空了，活著也不知道有什麼意義。 其實抗壓性很大，也很堅強，但這些堅強就像紙片一樣堆疊起來，輕輕一碰就倒了。 事實證明，只要有重大的變故，那些堅強，真的不夠用。 但願我有好幾顆心臟能承受，這些憂鬱又黑暗的心情。 明天，我還得要，用那制式的笑容面對大家，要假裝這一切沒發生過。"]
example_result2 = classifier_model(tf.constant(examples2))

print_my_examples(examples1, example_result1)
print_my_examples(examples2, example_result2)

# save the results

serving_results = reloaded_model.signatures['serving_default'](tf.constant(examples1))
serving_results = tf.sigmoid(serving_results['classifier'])

print_my_examples(examples1, serving_results)