import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def validate(model, dataset, args):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            outputs = model(inputs).numpy()
            outputs = [1 if o >= 0.5 else 0 for o in outputs]
            y_true.append(targets.numpy())
            y_pred.append(outputs)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print("\nValidation Set Performance")
    print('{:<15s} : {:5.6f}'.format("Balanced Acc.", balanced_accuracy_score(y_true, y_pred)))
    print('{:<15s} : {:5.6f}'.format("ROC AUC", roc_auc_score(y_true, y_pred)))
    print()


def validate_baseline(clf, X, y, clf_name):
    print()
    print(clf_name)
    print("-"*26)
    print("Validation Set Performance")
    print('{:<15s} : {:5.6f}'.format("Balanced Acc.", balanced_accuracy_score(y, clf.predict(X))))
    print('{:<15s} : {:5.6f}'.format("ROC AUC", roc_auc_score(y, clf.predict(X))))
    print()
