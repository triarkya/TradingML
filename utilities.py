from collections import defaultdict


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