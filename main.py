import argparse
from datetime import datetime

import baseline
import evaluation
import models.nn
import preprocessing
from engine import train


def app():
    print("FraudNet")
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessing.partition()
    train_dataset = preprocessing.Dataset(X_train, y_train)
    val_dataset = preprocessing.Dataset(X_val, y_val)

    clf = baseline.NaiveClassifier()
    clf.fit(X_train, y_train)
    evaluation.validate_baseline(clf, X_val, y_val)

    model = models.nn.FeedForwardNeuralNetwork(X_train.shape[1], args.hidden_dim, 3, args.batch_norm)

    try:
        t1 = datetime.now().timestamp()
        train(model, train_dataset, val_dataset, args)
        t2 = datetime.now().timestamp()
        time(t1, t2)
    except KeyboardInterrupt:
        print("Exiting Early")


def time(t1, t2):
    hours, rem = divmod(t2 - t1, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time: ""{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--hidden_dim', type=int, default=64)
    args = parser.parse_args()
    app()
