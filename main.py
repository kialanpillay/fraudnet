import argparse
import os.path
from datetime import datetime
from functools import partial

from ray import tune
from ray.tune import CLIReporter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import baseline
import engine
import evaluation
import preprocessing


def app():
    X, y = preprocessing.load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessing.partition(X, y)

    if args.baseline:
        clf = make_pipeline(StandardScaler(), baseline.NaiveClassifier())
        t1 = datetime.now().timestamp()
        clf.fit(X_train, y_train)
        metrics = evaluation.validate_baseline(clf, X_val, y_val)
        printer("Naive Classifier", "Validation", metrics)
        t2 = datetime.now().timestamp()
        time(t1, t2)

        svc = make_pipeline(StandardScaler(), LinearSVC(max_iter=1000))
        t1 = datetime.now().timestamp()
        svc.fit(X_train, y_train.ravel())
        t2 = datetime.now().timestamp()
        time(t1, t2)
        metrics = evaluation.validate_baseline(svc, X_val, y_val)
        printer("Support Vector Classifier", "Validation", metrics)

    if args.train:
        try:
            t1 = datetime.now().timestamp()
            if args.hyper_opt:
                config = {
                    "num_epochs": args.num_epochs,
                    "input_dim": X.shape[1],
                    "lr": tune.loguniform(1e-4, 1e-1),
                    "weight_decay": tune.loguniform(1e-5, 1e-3),
                    "batch_size": tune.grid_search([16, 32, 64]),
                    "batch_norm": tune.grid_search([True, False]),
                    "hidden_dim": tune.grid_search([16, 32, 64]),
                    "hidden_layers": tune.grid_search([3, 5, 7]),
                }
                filepath = os.path.join(os.getcwd(), 'data/creditcard.csv')
                reporter = CLIReporter(metric_columns=["loss", "balanced_accuracy", "iteration"])
                analysis = tune.run(partial(engine.train, filepath=filepath), config=config,
                                    progress_reporter=reporter)
                best_trial = analysis.get_best_trial("balanced_accuracy", "max", "last")
                print("Best Trial Config : {}".format(best_trial.config))
                print('{:<15s} : {:5.6f}'.format("Validation Loss", best_trial.last_result["loss"])),
                print('{:<15s} : {:5.6f}'.format("Balanced Acc.", best_trial.last_result["balanced_accuracy"]))

            else:
                config = {
                    "hyper_opt": args.hyper_opt,
                    "num_epochs": args.num_epochs,
                    "input_dim": X.shape[1],
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "batch_norm": args.batch_norm,
                    "hidden_dim": args.hidden_dim,
                    "hidden_layers": args.hidden_layers,
                }
                engine.train(config, verbose=args.verbose)
            t2 = datetime.now().timestamp()
            time(t1, t2)
        except KeyboardInterrupt:
            print("Exiting Early")


def printer(clf_name, partition, metrics):
    print("\n{}".format(clf_name))
    print("-" * 26)
    print("{} Set Performance".format(partition))
    print('{:<15s} : {:5.6f}'.format("Balanced Acc.", metrics['balanced_acc']))
    print('{:<15s} : {:5.6f}'.format("ROC AUC", metrics['roc_auc_score']))


def time(t1, t2):
    hours, rem = divmod(t2 - t1, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time: ""{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == "__main__":
    print("FraudNet")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--hyper_opt', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--hidden_layers', type=int, default=3)
    args = parser.parse_args()
    app()
