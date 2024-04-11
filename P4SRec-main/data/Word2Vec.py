# # -*- coding = utf-8 -*
# # @Time：2023/12/26 14:09
# # @Author:WS
# # @File:Word2Vec.py
# # @Software:PyCharm

from utils import _load_train_data_list
from gensim.models import Word2Vec


test_data_file = '../data/Toys.txt'
train_data_list= _load_train_data_list(data_file=test_data_file)

#print(train_data_list)
# 训练Word2Vec模型
model = Word2Vec(train_data_list, vector_size=100, window=5, min_count=1, workers=4)

# 保存训练好的模型
model.save('Word2vec_Toys.bin')

