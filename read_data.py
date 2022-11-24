from brat_parser import get_entities_relations_attributes_groups
import pandas as pd
from collections import defaultdict
import os
import glob


def read_ann(path):

    entities, relations, attributes, groups = get_entities_relations_attributes_groups(path)

    # Just focus on entities for now
    entities_df = pd.DataFrame.from_dict(entities, orient="index")

    return entities_df


def read_all_ann():
    path = r'/path-to-ann-files-here/'  # use your path
    all_files = glob.glob(path + "*/*/*.ann")

    # empty dict to hold dfs
    dfs = {}

    # for each file in the path above ending .ann ...
    for file in all_files:
        entity_df = read_ann(file)
        file_id = os.path.basename(os.path.dirname(file)) + "/" + os.path.basename(file).split('.')[0]
        dfs[file_id] = entity_df

    return dfs


def read_all_raw():
    path = r'/path-to-raw-files-here'  # use your path
    all_files = glob.glob(path + "*.txt")

    results = defaultdict(list)

    # for each file in the path above ending .ann ...
    for file in all_files:
        with open(file, "r") as file_open:
            results["filename"].append(os.path.basename(file).split('.')[0])
            results["text"].append(file_open.read())

    raw_files = pd.DataFrame(results)

    return raw_files
