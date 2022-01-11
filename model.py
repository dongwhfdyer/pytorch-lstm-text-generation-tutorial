import torch
from torch import nn

class Model(nn.Module):
    # 模型含有lstm 和embedding 层
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3
        # 存储特殊词汇的数量
        n_vocab = len(dataset.uniq_words)
        # 嵌入层传参，词汇数量和嵌入层维度
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        # lstm的输入和隐藏层，dropout 参数为0.2
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        # 全连接层的输出的个数输入是一致的，都是n_vocab
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        # 输出和状态是lstm的输出
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)

        return logits, state

    # 这个函数是为了初始化状态
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))
