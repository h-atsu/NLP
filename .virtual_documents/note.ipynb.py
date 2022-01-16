from utils import *
from datasets import ptb
import numpy as np


window_size = 2
wordvec_size = 100 #次元圧縮するサイズ
corpus,word_to_id, id_to_word = ptb.load_data('train')


vocab_size = len(word_to_id)


vocab_size


C = create_co_matrix(corpus, vocab_size, window_size)


W = ppmi(C, verbose=True)


from sklearn.utils.extmath import randomized_svd


U,S,V = randomized_svd(W, n_components=wordvec_size, n_iter = 5, random_state = None)


word_vecs = U[:, :wordvec_size]


querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_sililar(query, word_to_id, id_to_word, word_vecs, top=5)



