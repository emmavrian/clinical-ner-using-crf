"""Initially fetched from sklearn crfsuite tutorial and changed to fit project
https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html"""


# FEATURE ENGINEERING FUNCTIONS INCLUDING LEMMA
def word2features_withlemma(note, i):
    word = note[i][0]
    postag = note[i][1]
    lemma = note[i][2]

    features = {
        'bias': 1.0,                            # captures proportion of label in training set
        'word.lower()': word.lower(),           # word in lowercase
        'word[-3:]': word[-3:],                 # last three char of word
        'word[-2:]': word[-2:],                 # last two char of word
        'word.isupper()': word.isupper(),       # is word all uppercase?
        'word.istitle()': word.istitle(),       # does word start with uppercase, rest lowercase?
        'word.isdigit()': word.isdigit(),       # is word a number?
        'postag': postag,                       # postag
        'postag[:2]': postag[:2],               # first two char of postag
        'lemma': lemma,                         # lemma
    }
    if i > 0:
        word1 = note[i - 1][0]
        postag1 = note[i - 1][1]
        lemma1 = note[i - 1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:lemma': lemma1,
        })
    else:
        features['BOS'] = True                      # indicates that it is the beginning of the document

    if i < len(note)-1:
        word1 = note[i + 1][0]
        postag1 = note[i + 1][1]
        lemma1 = note[i + 1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:lemma': lemma1,
        })
    else:
        features['EOS'] = True                      # indicates that it is the end of the document

    return features


def note2features_withlemma(note):
    return [word2features_withlemma(note, i) for i in range(len(note))]


def note2labels_withlemma(note):
    return [label for token, postag, lemma, label in note]


def note2tokens_withlemma(note):
    return [token for token, postag, lemma, label in note]


def note2pos_withlemma(note):
    return [postag for token, postag, lemma, label in note]


# FEATURE ENGINEERING FUNCTIONS WITHOUT LEMMA
def word2features(note, i):
    word = note[i][0]
    postag = note[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = note[i - 1][0]
        postag1 = note[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(note)-1:
        word1 = note[i + 1][0]
        postag1 = note[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def note2features(note):
    return [word2features(note, i) for i in range(len(note))]


def note2labels(note):
    return [label for token, postag, label in note]


def note2tokens(note):
    return [token for token, postag, label in note]


def note2pos(note):
    return [postag for token, postag, label in note]
