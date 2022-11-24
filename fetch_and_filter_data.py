from read_data import read_all_raw, read_all_ann
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def fetch_and_filter_annotated_data():
    complete_raw_annotated_data = read_all_ann()

    relevant_types = ["Perifert_venekateter$", "Varm$", "Roed$", "Hoven$", "Smerte$",
                      "Pasient$", "Generelt_IV$", "Pus$", "IV_medisiner$", "IV_antibiotika$",
                      "IV_vaeske$", "IV_vaesker$", "IV_cellegift$", "IV_naeringsstoff$", "Hodepine$",
                      "Svimmelhet$", "Oemhet$", "Kvalme$", "Intravenoest_infusjon$", "Blodtrykk$", "Puls$",
                      "Kroppstemperatur$", "Respirasjons_frekvens$", "Bloedning$", "Kloeende$", "Pumpe$",
                      "Subkutan$"]

    # Combination of labels
    ivs = ["Generelt_IV", "IV_medisiner", "IV_antibiotika", "IV_vaeske", "IV_vaesker", "IV_cellegift",
           "Subkutan", "Pumpe", "IV_naeringsstoff", "Intravenoest_infusjon"]

    descriptive_signs = ["Hoven", "Pus", "Roed", "Smerte", "Varm", "Hodepine", "Svimmelhet", "Oemhet",
                         "Kvalme", "Bloedning", "Kloeende"]

    vital_signs = ["Blodtrykk", "Puls", "Kroppstemperatur", "Respirasjons_frekvens"]


    pattern = '|'.join(relevant_types)

    # empty list to hold dataframes filtered for only relevant types in extracted data
    complete_filtered_data = {}

    # iterate through all annotated files
    for filename, raw_ann_data in complete_raw_annotated_data.items():

        # don't include empty annotated files
        if raw_ann_data.empty:
            continue

        # only keep types that are in our relevant type list
        filtered_data = raw_ann_data[raw_ann_data['type'].str.contains(pattern)]

        # combination of labels
        filtered_data.loc[filtered_data['type'].isin(ivs), 'type'] = "IV_generelt"
        filtered_data.loc[filtered_data['type'].isin(descriptive_signs), 'type'] = "Tegn_Symptom"
        filtered_data.loc[filtered_data['type'].isin(vital_signs), 'type'] = "Vitalt_Tegn"

        # only have one tuple of span, so remove the room for extra tuples
        # NOTE: remove if this somehow changes in future
        filtered_data['span'] = filtered_data['span'].apply(lambda x: x[0])

        if not filtered_data.empty:
            complete_filtered_data[filename] = filtered_data.reset_index()

    return complete_filtered_data


def fetch_and_filter_raw_data():
    # read all raw data
    complete_raw_data = read_all_raw()
    # fetch filtered annotated data
    complete_filtered_annotated_data = fetch_and_filter_annotated_data()

    annotated_file_names = complete_filtered_annotated_data.keys()

    # only include raw data that has an annotated file, do not include if annotated file is empty
    temp = [v[6:] for v in annotated_file_names]  # find base names without folder
    temp = set(temp)  # convert to set to get unique values
    annotated_file_names_bases = list(temp)
    filtered_raw_data = complete_raw_data.loc[complete_raw_data['filename'].isin(annotated_file_names_bases)].copy()
    filtered_raw_data.reset_index()

    return filtered_raw_data


def fetch_full_data():
    annotated_data_dict = fetch_and_filter_annotated_data()
    raw_data_df = fetch_and_filter_raw_data()

    full_data = {}

    for index, row in raw_data_df.iterrows():
        raw_filename = row['filename']
        raw_text = row['text']

        # remove problematic unicode space breaks
        raw_text = raw_text.replace(u'\xa0', ' ')

        # Create list of annotated files from different annotators with same filename
        ann_keys = [k for k, v in annotated_data_dict.items() if k[6:] == raw_filename]
        # Create list of ann_dfs with same filename
        ann_dfs = [v for k, v in annotated_data_dict.items() if k in ann_keys]
        # Group dfs only based on span and type since that is all we need
        ann_dfs_grouped = [ann_df[["span", "type"]] for ann_df in ann_dfs]

        # initial value in final dict for current filename: just raw file
        full_data[raw_filename] = {"raw": raw_text}

        # Iterate over each row in each ann file and create dicts with span and type to add to final dict
        ann_number = 1
        for ann_df in ann_dfs_grouped:
            ann_lists = []
            for index, rows in ann_df.iterrows():
                # Create list for the current row
                row = [rows.span[0], rows.span[1], rows.type]
                # append the list to the final list
                ann_lists.append(row)
            # add the ann file to the dict for current filename
            full_data[raw_filename].update({"ann_"+str(ann_number): ann_lists})
            ann_number += 1

    return full_data
