from typing import Tuple
from train import Seq2Seq
import numpy as np
import torch
import pickle
import os

MAX_SEQUENCE_LENGTH = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)


def load_model(dirpath, model_ckpt="model.ckpt"):
    with open(os.path.join(dirpath, "src_lang.pickle"), "rb") as fi:
        src_lang = pickle.load(fi)
    with open(os.path.join(dirpath, "trg_lang.pickle"), "rb") as fi:
        trg_lang = pickle.load(fi)
    model = Seq2Seq.load_from_checkpoint(
        os.path.join(dirpath, model_ckpt),
        src_lang=src_lang,
        trg_lang=trg_lang,
    ).to(device)
    return model


# --------- PLEASE FILL THIS IN --------- #
def predict(functions: str,):
    batch_size = 128
    model = load_model(dirpath='models/best', model_ckpt="model.ckpt")
    prd_sentences, _, _ = model.predict(functions, batch_size=batch_size)
    return prd_sentences
# ----------------- END ----------------- #


def main(filepath: str = "test.txt"):
    """load, inference, and evaluate"""
    functions, true_derivatives = load_file(filepath)
    predicted_derivatives = predict(functions)
    scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
    print(np.mean(scores))


if __name__ == "__main__":
    main()
