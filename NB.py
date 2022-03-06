import pandas as pd
import seaborn as sns
from copy import deepcopy
import string
import numpy as np
import matplotlib.pyplot as plt

# Calculate the macro_f1 score
def get_macro_f1(true_labels, pred_labels, label_num):
    f1_sum = 0
    for label in range(label_num):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(true_labels)):
            if true_labels[i] == label:
                if pred_labels[i] == label:
                    tp += 1
                else:
                    fn += 1
            else:
                if true_labels[i] == label:
                    fp += 1
                else:
                    tn += 1
        f1 = 2 * tp / (2 * tp + fp + fn)
        f1_sum += f1
    return f1_sum / label_num

# Draw the confusion matrix
def get_confusion_matrix(true, pred, label_num, title):
    matrix = [[0 for _ in range(label_num)] for _ in range(label_num)]
    for i in range(len(true)):
        matrix[true[i]][pred[i]] += 1
    sns.set()
    sns.heatmap(matrix, annot=True, fmt=".5g")
    plt.title(title)
    plt.show()


# Save the prediction results to a file
def save_result(ori_data, pred_data, filename):
    result = {"SentenceId": list(ori_data["SentenceId"]), "Sentiment": pred_data}
    result = pd.DataFrame(result)
    file_path = "./" + filename
    result.to_csv(file_path, sep='\t', index=False)

# Reduce 5 labels to 3 labels
def reduce_label_num(data):
    new_data = deepcopy(data)
    map_dict = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
    for i in range(new_data.shape[0]):
        new_data.loc[i, "Sentiment"] = map_dict[new_data.loc[i, "Sentiment"]]
    return new_data


class NB_5:
    def __init__(self, train):
        self.label_num = 5  # set num of labels to 5
        self.train_labels = list(train["Sentiment"])  # get true label

        self.train_data = [words.split(" ") for words in list(train["Phrase"])]
        vocab = set([]) # It will include all the words in the training data with no duplicate words
        for words in self.train_data:
            vocab = vocab | set(words)
        self.vocab = list(vocab)  # create vocab of all words
        print("Training...")
        self.train()
        print("Finish!")

    # The input parameters of this function are the words in a whole sentence
    # Record the subscript positions of these words in vocab and store them in a vector (list)
    # If there are duplicate words in the sentence, the corresponding value will be increased, and then return the list
    def _words_in_vocab(self, words):
        # Iterate to see if the word appears, and increase the value if the word appears
        words_in_vocab = [0] * len(self.vocab)
        for word in words:
            if word in self.vocab:
                words_in_vocab[self.vocab.index(word)] += 1
        return words_in_vocab

    def train(self):
        # p(y|x) = p(x|y)*p(y)
        word_matrix = []  # for every sentence in data, store vector of word in vocab
        for data in self.train_data:
            word_matrix.append(self._words_in_vocab(data))
        words_count = len(word_matrix[0])  # number of all words
        data_count = len(word_matrix)  # number of sentences

        p_label_list = [0 for _ in range(self.label_num)]  # p(y)
        for label in self.train_labels:
            p_label_list[label] += 1
        self.p_label_list = [label_num / data_count for label_num in p_label_list]

        word_distribution = [np.ones(words_count) for _ in
                             range(self.label_num)]  # The words distribution for every label
        word_sum = [2.0 for _ in range(self.label_num)]  # The words sum for every label
        for i in range(data_count):
            word_distribution[self.train_labels[i]] += word_matrix[i]
            word_sum[self.train_labels[i]] += sum(word_matrix[i])
        self.p_word_in_label = [np.log(word_distribution[i] / word_sum[i]) for i in range(self.label_num)]  # p(x|y)

    def predict(self, pred):
        pred_data = [i.split() for i in list(pred["Phrase"])]
        pred_data_in_vocab = [np.array(self._words_in_vocab(dev_sen)) for dev_sen in pred_data]
        pred_label = []
        for i in pred_data_in_vocab:
            result = []
            for label in range(self.label_num):
                result.append(sum(i * self.p_word_in_label[label]) + np.log(self.p_label_list[label]))
            pred_label.append(result.index(max(result)))
        return pred_label


class NB_3(NB_5):
    def __init__(self, train):
        self.label_num = 3  # set num of labels to 3
        self.train_labels = list(train["Sentiment"])

        self.train_data = [words.split(" ") for words in list(train["Phrase"])]
        vocab = set([])
        for words in self.train_data:
            vocab = vocab | set(words)
        self.vocab = list(vocab)
        print("Training...")
        self.train()
        print("Finish!")


class NB_5_with_process(NB_5):
    def __init__(self, train, punctuations):
        self.label_num = 5  # set num of labels to 5
        self.train_labels = list(train["Sentiment"])

        self.train_data = []
        for words in list(train["Phrase"]):
            words = words.split(" ")
            processed_words = []
            for word in words:
                word = word.lower()
                if word not in punctuations:
                    processed_words.append(word)
            self.train_data.append(processed_words)

        vocab = set([])
        for words in self.train_data:
            vocab = vocab | set(words)
        self.vocab = list(vocab)
        print("Training...")
        self.train()
        print("Finish!")

    def _words_in_vocab(self, words):
        # Iterate through to see if the word appears, and if it does, set the word to 1
        words_in_vocab = [0] * len(self.vocab)
        for word in words:
            if word in self.vocab:
                words_in_vocab[self.vocab.index(word)] = 1
        return words_in_vocab


class NB_3_with_process(NB_5_with_process):
    def __init__(self, train, punctuations):
        self.label_num = 3  # set num of labels to 3
        self.train_labels = list(train["Sentiment"])

        self.train_data = []
        for words in list(train["Phrase"]):
            words = words.split(" ")
            processed_words = []
            for word in words:
                word = word.lower()
                if word not in punctuations:
                    processed_words.append(word)
            self.train_data.append(processed_words)

        vocab = set([])
        for words in self.train_data:
            vocab = vocab | set(words)
        self.vocab = list(vocab)
        print("Training...")
        self.train()
        print("Finish!")


if __name__ == '__main__':
    # read data
    train_data = pd.read_csv("./moviereviews/train.tsv", sep="\t")
    dev_data = pd.read_csv("./moviereviews/dev.tsv", sep="\t")

    print("NB_5:")
    nb = NB_5(train_data)
    dev_pred_result = nb.predict(dev_data)
    dev_true_result = list(dev_data["Sentiment"])
    NB_5_f1 = get_macro_f1(dev_true_result, dev_pred_result, 5)
    print("macro f1 score of data with 5 labels: ", NB_5_f1)
    # draw NB_5 confusion matrix
    get_confusion_matrix(dev_true_result, dev_pred_result, 5, "NB_5")

    train_data_3 = reduce_label_num(train_data)
    dev_data_3 = reduce_label_num(dev_data)

    print("NB_3:")
    nb_3 = NB_3(train_data_3)
    dev_pred_result_3 = nb_3.predict(dev_data_3)
    dev_true_result_3 = list(dev_data_3["Sentiment"])
    NB_3_f1 = get_macro_f1(dev_true_result_3, dev_pred_result_3, 3)
    print("macro f1 score of data with 3 labels: ", NB_3_f1)
    # draw NB_3 confusion matrix
    get_confusion_matrix(dev_true_result_3, dev_pred_result_3, 3, "NB_3")

    punctuations = string.punctuation

    punctuations += '...'
    punctuations += '--'
    punctuations += '``'
    punctuations += "''"

    print("NB_5_with_process:")
    nb_wp = NB_5_with_process(train_data, punctuations)
    dev_pred_result_wp = nb_wp.predict(dev_data)
    NB_5_wp_f1 = get_macro_f1(dev_pred_result_wp, dev_true_result, 5)
    print("after preprocess, macro f1 score of data with 5 labels: ", NB_5_wp_f1)
    # draw NB_5_wp confusion matrix
    get_confusion_matrix(dev_true_result, dev_pred_result_wp, 5, "NB_5_wp")
    save_result(dev_data, dev_pred_result_wp, "./dev_predictions_5classes_Ruiqing_Xu.tsv")

    print("NB_3_with_process:")
    nb_wp_3 = NB_3_with_process(train_data_3, punctuations)
    dev_pred_result_wp_3 = nb_wp_3.predict(dev_data_3)
    NB_3_wp_f1 = get_macro_f1(dev_true_result_3, dev_pred_result_wp_3, 3)
    print("after preprocess, macro f1 score of data with 3 labels: ", NB_3_wp_f1)
    # draw NB_3_wp confusion matrix
    get_confusion_matrix(dev_true_result_3, dev_pred_result_wp_3, 3, "NB_3_wp")
    save_result(dev_data, dev_pred_result_wp_3, "./dev_predictions_3classes_Ruiqing_Xu.tsv")

    test_data = pd.read_csv("./moviereviews/test.tsv", sep="\t")
    test_pred_result = nb_wp.predict(test_data)
    save_result(test_data, test_pred_result, "./test_predictions_5classes_Ruiqing_Xu.tsv")
    test_pred_result_3 = nb_wp_3.predict(test_data)
    save_result(test_data, test_pred_result_3, "./test_predictions_3classes_Ruiqing_Xu.tsv")
