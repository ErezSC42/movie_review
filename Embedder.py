import os
import numpy as np
from gensim.parsing.preprocessing import strip_tags,strip_multiple_whitespaces,strip_non_alphanum


class Embedder():
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
                pad_count = self.sent_len - len(item)
                if pad_count > 0: #we need to pad with zeros
                    pad_em = np.zeros([50,pad_count]).T
                    em = np.vstack([em,pad_em]).astype("double")
                return em.T
            return self.embeddings[self.word_dict[item]]
        pass

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

    def __init__(self,parser,word_embedding_len,sentance_len,pre_trained_model_path):
        self.sentance_len = sentance_len  # will use to determine the matrix dim
        self.word_embedding_len = word_embedding_len
        self.pretrained_model_path = os.path.join(os.path.dirname(__file__), pre_trained_model_path)
        words,embeddings = self.load_text_embedding(self.pretrained_model_path)
        word_dict = {word: index for index, word in enumerate(words)}
        self.lang_model = self.GloveLangModel(embeddings,word_dict,sentance_len)

    def str_to_image(self,input_str):
        preprocessed_str = strip_multiple_whitespaces(strip_non_alphanum(strip_tags(input_str))).lower().split()
        N = len(preprocessed_str)
        output_image = None
        if N >= self.sentance_len:
            preprocessed_str = preprocessed_str[:self.sentance_len]
        return self.lang_model[preprocessed_str].T

    def str_series_to_image(self,series_str):
        DATA_LEN = len(series_str)
        output_image = np.zeros([DATA_LEN,self.sentance_len,self.word_embedding_len])
        print(DATA_LEN)
        for i in range(DATA_LEN):
            output_image[i,:,:] = self.str_to_image(series_str[i])
        return output_image




