from crf_feature_engineering import note2features, note2labels, note2pos, note2features_withlemma, note2labels_withlemma, note2pos_withlemma
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter

def crf_evaluation(crfsuite_data, lemma: bool):

    # Feature engineering
    if(lemma):
        X = [note2features_withlemma(n) for n in crfsuite_data]
        y = [note2labels_withlemma(n) for n in crfsuite_data]
        pos = [note2pos_withlemma(n) for n in crfsuite_data]
    else:
        X = [note2features(n) for n in crfsuite_data]
        y = [note2labels(n) for n in crfsuite_data]
        pos = [note2pos(n) for n in crfsuite_data]

    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, shuffle=True)

    # Create a CRF model with sklearn_crfsuite
    # Using L-BFGS training algorithm (it is default) with Elastic Net (L1 + L2) regularization.
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    # Train the CRF model using the training data
    crf.fit(X_train, y_train)

    ### Evaluation ###

    # remove O from labels when calculating metrics since it is very overrepresented
    labels = list(crf.classes_)
    labels.remove('O')

    # predict labels in test set
    y_pred_test = crf.predict(X_test)

    # predict labels in train set (for comparison)
    y_pred_train = crf.predict(X_train)

    # group B and I results
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    if(lemma):
        print("\nMETRICS WITH LEMMA:")
    else:
        print("\nMETRICS NO LEMMA:")

    # obtaining metrics on the test set
    print('Test set classification report: \n\n{}'.format(
        metrics.flat_classification_report(y_test, y_pred_test, labels=sorted_labels
    )))

    # obtaining metrics on the train set
    print('Train set classification report: \n\n{}'.format(metrics.flat_classification_report(
        y_train, y_pred_train, labels=sorted_labels
    )))

    print("\nTransitions:\n")

    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    print("Top likely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common(20))
    print("\nTop unlikely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common()[-20:])

    print("\nState features:\n")

    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    print("Top positive:")
    print_state_features(Counter(crf.state_features_).most_common(30))

    print("\nTop negative:")
    print_state_features(Counter(crf.state_features_).most_common()[-30:])

    print("\n#########################\n")

    return y, pos
