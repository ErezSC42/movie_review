from gensim.models import Word2Vec,KeyedVectors
from Embedder import Embedder
import os
import numpy as np

MY_DICT_PATH = r"C:\Users\erezs\PycharmProjects\RL_PROJECT\glove.6B.50d.txt"
#this class implemets the word embedding feature
#the main purpose is to convert the game state parse string to image
#sentance, which  is a string will be converted to a 2d numpy array with the following dim:
#[word1,word2,word3...,wordN] -> numpy.array([300,N]) where N == sentance_len
class GloveEmbedder(Embedder):

    class GloveLangModel():
        def __init__(self,embeddings,word_dict,sent_len):
            self.embeddings = embeddings
            self.word_dict = word_dict
            self.sent_len = sent_len

        def __getitem__(self, item):
            if type(item) == list:
                em = []
                for x in item:
                    try:
                        em.append(self.embeddings[self.word_dict[x]].T)
                    except:
                        em.append(self.embeddings[self.word_dict["unknown"]].T)
                em = np.array(em)
                #em = np.array([self.embeddings[self.word_dict[x]].T for x in item ])
                pad_count = self.sent_len - len(item)
                if pad_count > 0: #we need to pad with zeros
                    pad_em = np.zeros([pad_count,50])
                    em = np.vstack([em,pad_em]).astype("double")
                return em
            return self.embeddings[self.word_dict[item]]
        pass


    def __init__(self,parser,word_embedding_len,sentance_len,pre_trained_model_path=MY_DICT_PATH):
        super(GloveEmbedder,self).__init__(parser,word_embedding_len,sentance_len)
        self.pretrained_model_path = os.path.join(os.path.dirname(__file__), pre_trained_model_path)
        words,embeddings = self.load_text_embedding(MY_DICT_PATH)
        word_dict = {word: index for index, word in enumerate(words)}
        self.lang_model = self.GloveLangModel(embeddings,word_dict,sentance_len)


    def load_text_embedding(self,path):
        """
        Load any embedding model written as text, in the format:
        word[space or tab][values separated by space or tab]
        :return: a tuple (wordlist, array)
        """
        words = []
        # start from index 1 and reserve 0 for unknown
        vectors = []
        with open(path, 'rb') as f:
            for line in f:
                line = line.decode('utf-8')
                line = line.strip()
                if line == '':
                    continue

                fields = line.split(' ')
                word = fields[0]
                words.append(word)
                vector = np.array([float(x) for x in fields[1:]], dtype=np.float32)
                vectors.append(vector)

        embeddings = np.array(vectors, dtype=np.float32)
        return words, embeddings



