# Derivatives Calculation Using Seq2seq Transformer
This assignment utlizes the Seq2seq transformer to learns the derivative of the function with respect to the requested variable. 
- This repo tests the model against a toy dataset of derivative calculation
- The model `Seq2Seq` top layer module is implemented using PyTorch Lightning
- The python environment is python 3.7 and the all python dependencies and their versions are included in (`requirements.txt`)


## Data and preprecess
- The full dataset (`data/data.txt`) contains a million examples.
- The full dataset is split into the train set (`data/train_set.txt`) and test set (`data/test_set.txt`) using `data_preprocess.py` 

## Training and Evaluation

Train the model (use `--help` for more options):
```shell
python train.py \
    "models/best" \
    --gpus 1 \
    --gradient_clip_val 1 \
    --max_epochs 10 \
    --val_check_interval 0.2
```
After the training, the final score on test set (a portion of the train.txt) is saved in (`models/best/eval.txt`). The model accuracy is `96.88%` with training on 10 epochs.
The trained model is stored as (`models/best/model.ckpt`). 

## Brief explanation
The derivative calculation process is similar to machine translation tasks in NLP, and both of then can be considered as sequence to sequence model. Therefore, we choose to utilize the `seq2seq transformer` model, which is well-performed in machine translation tasks, to learn the derivative calculation rules and even chain rules.

Specifically, the seq2seq transformer model include a encode and a decoder. Given the function, the encoder encodes it into hidden representation and the decoder is responsible to transform it into derivatives with requested variables.

The number of trainable parameters for the whole model is 4M, and specific configurations at each layer are summarized in (`network.txt`)
