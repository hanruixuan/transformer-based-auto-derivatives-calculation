import random
import argparse


def train_test_split(pairs, train_test_split_ratio):
    random.shuffle(pairs)
    split = int(train_test_split_ratio * len(pairs))
    train_pairs, test_pairs = pairs[:split], pairs[split:]
    return train_pairs, test_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates train test split.")
    parser.add_argument("--datapath", type=str, default="data/train.txt")
    parser.add_argument("--trainpath", type=str, default="data/train_set.txt")
    parser.add_argument("--testpath", type=str, default="data/test_set.txt")
    parser.add_argument("--ratio", type=int, default=0.98)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.datapath) as f:
        pairs = f.read().splitlines()
        train_pairs, test_pairs = train_test_split(pairs, args.ratio)

    with open(args.trainpath, "w") as f:
        f.write("\n".join(train_pairs) + "\n")

    with open(args.testpath, "w") as f:
        f.write("\n".join(test_pairs) + "\n")

    print(f"num pairs: {len(pairs)}")
    print(f"num train pairs:{len(train_pairs)}")
    print(f"num test pairs:{len(test_pairs)}")
