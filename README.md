# Derivatives Calculation Using Seq2seq Transformer
This repository utilizes the Seq2seq transformer to learn the derivative of the function for the requested variable. 
- This repo tests the model against a toy dataset of derivative calculation
- The model `Seq2Seq` top layer module is implemented using PyTorch Lightning
- The Python environment is Python 3.7 and all Python dependencies and their versions are included in (`requirements.txt`)
- The derivatives equation for training should follow the format: "d(-cos^3(4e^7))/de=84e^6*sin(4e^7)*cos^2(4e^7)"

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

## Brief explanation
The derivative calculation process is similar to machine translation tasks in NLP, and both of them can be considered as sequence-to-sequence model. Therefore, the `seq2seq transformer` model, which is well-performed in machine translation tasks, is used to learn the derivative calculation rules and even chain rules.
