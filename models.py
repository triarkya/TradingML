from sklearn.ensemble import RandomForestClassifier


def setup_random_forest(parameters):
    return RandomForestClassifier(**parameters)
