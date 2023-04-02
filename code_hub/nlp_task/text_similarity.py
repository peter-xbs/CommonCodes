# _*_ coding:utf-8 _*_
# @Time: 2023-04-02 18:47
# @Author: peters
# @Email: xinbao.sun@hotmail.com
# @File: text_similarity.py
# @Project: CommonCodes

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len):
        self.data = [item[:max_len] for item in data]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
def encode_texts(texts):
    all_vecs = []
    # 加载模型
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained('/root/model/IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
    tokenizer = AutoTokenizer.from_pretrained('/root/model/IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
    net = torch.nn.DataParallel(model)
    model.to("cuda:0")

    def txt2emb(batch_txt):
        with torch.no_grad():
            inp = tokenizer(batch_txt, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
            inp.to("cuda:0")
            out = net(**inp, output_hidden_states=True)
            res = out.hidden_states[-1][:, 0, :].cpu().numpy()

        return res.tolist()

    dataset = Dataset(texts, max_len=512)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=False)
    for batch in dataloader:
        vecs = txt2emb(batch)
        all_vecs.extend(vecs)
    return all_vecs




