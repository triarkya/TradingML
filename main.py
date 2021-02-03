from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import utilities
import models


if __name__ == '__main__':

    forest = models.setup_random_forest()

    X, Y = utilities.read_dataset("BTCUSDT_300days_5MINUTE_ta.csv", ignore_class=None)
    split_size = int(len(X) * 0.75)

    X_train, X_test = X[:split_size], X[split_size:]
    Y_train, Y_test = Y[:split_size], Y[split_size:]

    run_tuning = False
    optimize_parameters = {
        'n_estimators': [100, 300, 500, 800, 1200],
        'max_depth': [5, 8, 15, 25, 30],
        'min_samples_split': [2, 5, 10, 15, 100],
        'min_samples_leaf': [1, 2, 5, 10]
    }

    if run_tuning:
        best_parameters = utilities.hyperparameter_classification_tuning(
            forest,
            optimize_parameters,
            (X_train, Y_train)
        )
    else:
        # from previous tuning
        best_parameters = {
            'max_depth': 30,
            'min_samples_leaf': 10,
            'min_samples_split': 5,
            'n_estimators': 100
        }

    # update random forest
    forest = models.setup_random_forest(best_parameters)
    model = forest.fit(X_train, Y_train)

    # print stats
    print('Correct Prediction (%): ', accuracy_score(Y_test, model.predict(X_test), normalize=True) * 100.0)
    print(classification_report(Y_test, model.predict(X_test)))