import pandas as pd
from fetch_and_filter_data import fetch_full_data
from crf_preprocessing import crf_preprocessing
from collections import Counter
import json
from crf_evaluation import crf_evaluation


def main():

    print("main")

    """FIRST: GENERATE CRFSUITE FRIENDLY DATASET FROM TXT AND ANN FILES"""

    """
    # fetch full data
    full_data = fetch_full_data()

    # convert to crfsuite friendly format
    crfsuite_data = crf_preprocessing(full_data)

    # save as json
    with open('crfsuite_data/crfsuite_data_no_lemma_both_sessions.json', 'w') as f:
        json.dump(crfsuite_data, f)

    """

    """CRF MODEL"""

    # read the files
    with open('crfsuite_data/crfsuite_data_lemma_both_sessions.json') as f:
        crfsuite_data_lemma_both_sessions = [list(x) for x in json.load(f)]

    with open('crfsuite_data/crfsuite_data_no_lemma_both_sessions.json') as f:
        crfsuite_data_no_lemma_both_sessions = [list(x) for x in json.load(f)]

    print("Total number of notes: ", len(crfsuite_data_no_lemma_both_sessions))

    y, pos = crf_evaluation(crfsuite_data_no_lemma_both_sessions, lemma=False)
    y_l, pos_l = crf_evaluation(crfsuite_data_lemma_both_sessions, lemma=True)

    # OVERVIEW OF COUNT IOB+LABELS
    list_of_all_individual_labels = []
    for note in y_l:
        for label in note:
            list_of_all_individual_labels.append(label)

    iob_values = Counter(list_of_all_individual_labels).values()  # counts the elements' frequency

    print("Total number of tokens:", sum(iob_values))

    df_iob_counts = pd.DataFrame.from_dict(Counter(list_of_all_individual_labels), orient='index').reset_index()
    df_iob_counts.columns = ["IOB tag", "count"]
    print(df_iob_counts)

    # OVERVIEW OF COUNT POS
    list_of_all_individual_pos = []
    for note in pos_l:
        for pos in note:
            list_of_all_individual_pos.append(pos)

    pos_values = Counter(list_of_all_individual_pos).values()  # counts the elements' frequency

    print("\nTotal number of tokens:", sum(pos_values))

    df_pos_counts = pd.DataFrame.from_dict(Counter(list_of_all_individual_pos), orient='index').reset_index()
    df_pos_counts.columns = ["POS tag", "count"]
    df_pos_counts = df_pos_counts.sort_values(by=["count"], ascending=False)
    print(df_pos_counts)

    # Print one note (first one)
    #print()
    #print(crfsuite_data_lemma_both_sessions[0])


if __name__ == "__main__":
    main()

