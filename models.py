from sklearn.ensemble import RandomForestClassifier


def setup_random_forest(parameters=None):
    if parameters is not None:
        return RandomForestClassifier(**parameters)
    else:
        return RandomForestClassifier()
