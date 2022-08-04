import collections
import re
from tabnanny import verbose
import numpy as np
from tqdm import tqdm


def preprocess(text):
    text = text.lower().replace('.', ' .')
    #text = re.split('\W+', text)
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            id = len(word_to_id)
            word_to_id[word] = id
            id_to_word[id] = word
    corpus = np.array([word_to_id[word] for word in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similality(x, y, eps=1e-8):
    nx = x / (np.linalg.norm(x) + eps)
    ny = y / (np.linalg.norm(y) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print("{} is not found".format(query))

    print("\n [query] " + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.array([cos_similality(query_vec, wvec)
                           for wvec in word_matrix])

    count = 0
    for i in (-similarity).argsort():
        if id_to_word[i] == query:
            continue
        print("{:10}: {:.5}".format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s is not found' % word)
            return

    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]
                                      ], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" +
              str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x


def ppmi(C, verbose=False, eps=1e-8):
    """
    comatrix express joint probability distribution 
    P(i,j) = C[i][j] 
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    bar = tqdm(total=total)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[i] * S[j]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                bar.update(1)

    return M


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size: -window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def one_hot_encoding(x, vocab_size):
    ohe = np.zeros(vocab_size)
    ohe[x] = 1
    return ohe


def convert_one_hot(data, vocab_size):
    """
    data : data to convert one hot (train set or target set)
    """
    N = data.shape[0]

    if data.ndim == 1:
        # for target data set
        one_hot = np.array([one_hot_encoding(x, vocab_size) for x in data])

    elif data.ndim == 2:
        # for training data set
        C = data.shape[1]  # context words length
        one_hot = np.array([[one_hot_encoding(x, vocab_size)
                             for x in xs] for xs in data])

    return one_hot


class UnigramSampler:
    def __init__(self, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter(corpus)

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        negative_sample = np.zeros(
            (batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(
                self.vocab_size, size=self.sample_size, replace=False, p=p)

        return negative_sample


if __name__ == '__main__':
    text = "you say goodbye and i say hello."
    corpus, w2i, i2w = preprocess(text)
    sampler = UnigramSampler(corpus)
    target = np.array([1, 3, 0])
    print(sampler.get_negative_sample(target))
