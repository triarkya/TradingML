from collections import defaultdict
import pandas as pd
from sklearn.model_selection import GridSearchCV


"""
read data from datafile and split into input and output data
X: input data
Y: output data (for example classification)
"""
def read_dataset(datafile, ignore_class=None):
    data = pd.read_csv(datafile)
    all_columns = data.columns

    if ignore_class is not None:
        data = data[data.classification != ignore_class]

    # divide dataset
    class_zero = data[data.classification == 0]
    class_rest = data[data.classification != 0]

    # reduce class zero and update full dataset
    reduced_zero = reduce_dataset_class(class_zero, 6)
    reduced_dataset = pd.concat([reduced_zero, class_rest], sort=False).sort_index()

    # classification = data["classification"].tolist()
    # reduced_classification = reduced_dataset["classification"].tolist()
    # print("Before:", classification.count(0), classification.count(1), classification.count(2))
    # print("After:", reduced_classification.count(0), reduced_classification.count(1), reduced_classification.count(2))

    x_cols = [
        "close",
        "volume",
        "ema_25",
        "vwma_50",
        "vwma_100",
        "vwma_200",
        "macd",
        "macds",
        "macdh",
        "mfi",
        "adx",
        "di_neg",
        "di_pos",
        "rvgi",
        "rvgi_signal"
    ]

    y_cols = [all_columns[-1]]

    X = reduced_dataset[x_cols]
    Y = reduced_dataset[y_cols]

    return X, Y.values.ravel()


"""
dataset_from_class: Pandas Dataframe of class to reduce
blocksize: how many blocks should each class block be 

example: dataset_from_class is Dataframe of class 0 and blocksize = 8
then each incrementing block will be only 8 candlesticks
"""
def reduce_dataset_class(dataset_from_class, blocksize):
    all_row_nums = dataset_from_class.index.tolist()

    # find all blocks
    blocks_dict = defaultdict(list)
    current = 0
    block_num = 0
    while current < len(all_row_nums):
        if len(blocks_dict[block_num]) == 0:
            # block has not started yet
            blocks_dict[block_num].append(all_row_nums[current])
            current += 1
        elif len(blocks_dict[block_num]) == 1:
            # block only has a start_value
            blocks_dict[block_num].append(all_row_nums[current])
            current += 1
        elif len(blocks_dict[block_num]) == 2:
            if blocks_dict[block_num][1] == all_row_nums[current] - 1:
                # current value is only one incremented to previous (same block)
                blocks_dict[block_num][1] += 1
                current += 1
            else:
                # init new block
                block_num += 1

    # reduce block intervals if necessary
    remaining_rows = []
    for block, block_interval in blocks_dict.items():
        while len(range(block_interval[0], block_interval[1] + 1)) > blocksize:
            blocks_dict[block][0] += 1
            blocks_dict[block][1] -= 1
        remaining_rows += list(range(blocks_dict[block][0], blocks_dict[block][1] + 1))

    # reduce class dataframe to remaining blocks only
    dataset_reduced = dataset_from_class[dataset_from_class.index.isin(remaining_rows)]
    return dataset_reduced


"""
do some hyperparameter tuning and return best parameters
model: something like RandomForestClassifier for example
parameters: dictionary with parameters to test
data: tuple with data[0] as input data and data[1] as output data
"""
def hyperparameter_classification_tuning(model, parameters, data):
    grid_tuning = GridSearchCV(
        model,
        parameters,
        cv=3,
        verbose=4,
        n_jobs=-1,
        scoring="balanced_accuracy"
    )

    grid_tuning.fit(data[0], data[1])

    return grid_tuning.best_params_
