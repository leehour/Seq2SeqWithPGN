# CarDialogSummary
Extract reports from conversations about cars.

## Introduction to each file

### 1.config.py
* define paths used in this project.

### 2.split_data.py
* split train and test data.
* save to train_seg.csv/test_seg.csv.

### 3.train_word_vec.py
* train word2vec model(if you have trained model, skip this step).
* merge input data and save to train_seg_merge.csv/test_seg_merge.csv.

### 4. seq2seq.model.py
* seq2seq model, which contains encoder, attention, decoder.

### 5.main.py
* load word2vec model.
* build training data.
* train and predict data.

## Running steps
1.run split_data.py, generate words that are splited  
2.run train_word_vec.py, generate the word2vec model  
3.run main.py, train model and predict test data

## Update
* Add switch to turn on bi-gru model.
* Add switch to turn on dropout.
* Change the optimization function from Adam to Adagrad, add clipping gradient.