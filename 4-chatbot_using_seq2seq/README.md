## 简介
基于 TensoFlow 构建 SeqSeq 模型，并加入 Attention 机制，encoder 和 decoder 为 3 层的 RNN 网络。本教程主要参考 TensorFlow 官网 translate Demo。
## 步骤简介

- generate_chat.py - 清洗数据、提取 ask 数据和 answer 数据、提取词典、为每个字生成唯一的数字 ID、ask 和 answer 用数字 ID 表示；
- seq2seq.py、seq2seq_model.py - TensorFlow 中 Translate Demo，由于出现 deepcopy 错误，这里对 SeqSeq 稍微改动了；
- train_chat.py - 训练 SeqSeq 模型；
- predict_chat.py - 进行聊天。

## 文件清单
|:---:|:---:|
|chatbot.ckpt开头的文件|训练好的模型|
|chat_model.zip|训练好的模型的zip包|
|chat.conv|原始语料|
|vocab_开头的文件|前期训练产生的词典文件|