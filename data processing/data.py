import pandas as pd
import numpy as np

cochlear_data_path = "data processing/data-cochlear.txt"


df = pd.read_csv(cochlear_data_path, delimiter="\t")
speech2vec = pd.read_csv('data processing/embeddings/speech2vec.txt', delimiter=" ", header=1)
word2vec = pd.read_csv("data processing/embeddings/word2vec.txt", delimiter=" ", header= 1)
word2vec = word2vec.T
word2vec.drop([0])



print(word2vec)
word2vec.to_csv("2134.csv")

# list_words = df["Word"].tolist()
# set_words = list(set(list_words)) 
