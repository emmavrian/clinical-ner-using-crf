import obt
import textspan


def crf_preprocessing(full_data):

    crfsuite_data = []

    for filename, data in full_data.items():

        # NOTE: CAN NOT DIVIDE BY INDIVIDUAL SENTENCES. TRIED THIS FIRST, BUT PROBLEM BECAUSE OF MATCHING THE
        # SPAN VALUES IN THE ANNOTATED NOTE TO THE ACTUAL NOTE

        raw = data["raw"]

        # list of annotated files only
        ann_files = [v for k, v in data.items() if k[:3] == "ann"]

        # list of words in raw and their POS tag by using obt_tagging function (norwegian Oslo-Bergen Tagger)
        pos_tagged_words = obt_tagging(raw)

        # create a list of words only based on the word segmentation from obt
        list_of_words = [w[0] for w in pos_tagged_words]  # list of all words

        # get original spans of list of words based on raw text. Input: list of words from pos tagging and raw note
        list_of_indices = textspan.get_original_spans(list_of_words, raw)

        # create list of indices and words
        list_of_indices = [x for b in list_of_indices for x in b]  # remove the unneccesary list [(0,0)] per element
        list_of_indices = [list(x) for x in list_of_indices]  # convert tuples to list
        list_of_words_and_index = [[a] + b for a, b in zip(list_of_words, list_of_indices)]

        # each annotated file will be its own note in the final dataset
        for ann in ann_files:

            # get list of words and corresponding IOB tags for ann1
            iob_tagged_words_ann = iob_tagging_ann(ann, list_of_words_and_index)

            # merge pos and iob for ann based on first element in lists (the word)
            merged_pos_iob_ann = merge(pos_tagged_words, iob_tagged_words_ann)

            # convert every element to tuple for final processing of sentence
            crfsuite_data_ann = [tuple(l) for l in merged_pos_iob_ann]

            # append data to final dataset
            crfsuite_data.append(crfsuite_data_ann)

    return crfsuite_data


def merge(lst1, lst2):
    """Merge two lists according to first element of list (here, the word)
    from: https://www.geeksforgeeks.org/python-merge-two-list-of-lists-according-to-first-element/"""
    return [a + [b[1]] for (a, b) in zip(lst1, lst2)]


def obt_tagging(raw):
    """OSLO-BERGEN TAGGER"""
    full_obt = obt.tag_bm(raw)
    obt_tagged_words = [[w["word"], "NN"] for w in full_obt]
    # also include lemma:
    # obt_tagged_words = [[w["word"], "NN", "lemma"] for w in full_obt]

    # for tagged_word in full_obt:
    for i in range(0, len(obt_tagged_words)):
        obt_tagged_words[i][1] = full_obt[i]["ordklasse"]
        # also include lemma:
        # obt_tagged_words[i][2] = full_obt[i]["base"]

    return obt_tagged_words


def iob_tagging_ann(ann, list_of_words_and_index):
    """ANNOTATOR BASED ON ANNOTATED NOTES AND IOB-METHOD"""
    # use list of words to create initial IOB list for ann with everything labeled as default "O"
    iob_tagged_words = [[w[0], w[1], w[2], "O"] for w in list_of_words_and_index]

    # note:
    # iob_tagged_words -> word[0] = word, word[1] = start of span, word[2] = end of span, word[3] = IOB label
    # annotation -> annotation[0] = start of span, annotation[1] = end of span, annotation[2] = annotated label
    for annotation in ann:
        multi_word_label = False
        for word in iob_tagged_words:
            # if start and end index of word is start and end index of annotated label
            if multi_word_label:
                # set IOB label to I + annotation label
                word[3] = "I-" + annotation[2]
                # check if this is end of multi_word_label (or if we're over the end in case of combined words)
                if word[2] == annotation[1] or word[2] > annotation[1]:
                    word[3] = "I-" + annotation[2]
                    break # break word for loop
            elif word[1] == annotation[0] and word[2] == annotation[1]:
                # set IOB label to B + annotation label
                word[3] = "B-" + annotation[2]
                break # break word for loop, done with current annotation
            # or if start index of word is start of annotated label only, not matching end
            elif word[1] == annotation[0] and word[2] != annotation[1]:
                word[3] = "B-" + annotation[2]
                multi_word_label = True

    final_iob_tagged_words = [[w[0], w[3]] for w in iob_tagged_words]

    return final_iob_tagged_words
