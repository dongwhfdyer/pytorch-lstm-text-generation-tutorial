import torch
import pandas as pd
from collections import Counter
# 数据处理部分最重要

# 首先继承了torch中的datesets的类
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.words = self.load_words()
        # 特殊词汇是什么
        self.uniq_words = self.get_uniq_words()
        # 建立由词语到索引的和由索引到词语的字典
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        # 词语所对应的索引列表
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        # 通过csv文件进行数据收集,这里返回的是单个的单词列表集合
        train_df = pd.read_csv('data/reddit-cleanjokes.csv')
        text = train_df['Joke'].str.cat(sep=' ')
        return text.split(' ')

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]),
        )


