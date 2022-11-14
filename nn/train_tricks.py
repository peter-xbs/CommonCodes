# _*_ coding:utf-8 _*_

"""
@Time: 2022/11/9 11:55 上午
@Author: jingcao
@Email: xinbao.sxb@alibaba-inc.com
"""

import torch

"""
对抗训练相关技巧: https://zhuanlan.zhihu.com/p/103593948
"""


class FGM(object):
    """
    REF: Adversarial training methods for semi-supervised text classification
    在训练时，只需要额外添加5行代码：
    已成功测试，可用
    fgm = FGM(model) # （#1）初始化
    for batch_input, batch_label in data:
        loss = model(batch_input, batch_label) # 正常训练
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # （#2）在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label) # （#3）计算含有扰动的对抗样本的loss
        loss_adv.backward() # （#4）反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # （#5）恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}  # 用于保存模型扰动前的参数

    def attack(self, epsilon=1., emb_name='word_embeddings'):  # emb_name表示模型中embedding的参数名):
        """
        生成扰动和对抗样本
        :param epsilon:
        :param emb_name:
        :return:
        """
        for name, param in self.model.named_parameters():  # 遍历模型的所有参数
            if param.requires_grad and emb_name in name:  # 只取word embedding层的参数
                self.backup[name] = param.data.clone()  # 保存参数值
                norm = torch.norm(param.grad)  # 对参数梯度进行二范式归一化
                if norm != 0 and not torch.isnan(norm):  # 计算扰动，并在输入参数值上添加扰动
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        """
        恢复添加扰动的参数
        :param emb_name:
        :return:
        """
        for name, param in self.model.named_parameters():  # 遍历模型的所有参数
            if param.requires_grad and emb_name in name:  # 只取word embedding层的参数
                assert name in self.backup
                param.data = self.backup[name]  # 重新加载保存的参数值
        self.backup = {}


class PGD(object):
    """
    TODO 本实现仍有瑕疵，无法使用，待有时间再进一步升级
    pgd = PGD(model)
    K = 3
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        pgd.backup_grad()
        # 累积多次对抗训练——每次生成对抗样本后，进行一次对抗训练，并不断累积梯度
    for t in range(K):
        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
        if t != K-1:
            model.zero_grad()
        else:
            pgd.restore_grad()
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    pgd.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()
"""


def __init__(self, model):
    self.model = model
    self.emb_backup = {}
    self.grad_backup = {}


def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
    for name, param in self.model.named_parameters():
        if param.requires_grad and emb_name in name:
            if is_first_attack:
                self.emb_backup[name] = param.data.clone()
            norm = torch.norm(param.grad)
            if norm != 0 and not torch.isnan(norm):
                r_at = alpha * param.grad / norm
                param.data.add_(r_at)
                param.data = self.project(name, param.data, epsilon)


def restore(self, emb_name='word_embeddings'):
    for name, param in self.model.named_parameters():
        if param.requires_grad and emb_name in name:
            assert name in self.emb_backup
            param.data = self.emb_backup[name]
    self.emb_backup = {}


def project(self, param_name, param_data, epsilon):
    r = param_data - self.emb_backup[param_name]
    if torch.norm(r) > epsilon:
        r = epsilon * r / torch.norm(r)
    return self.emb_backup[param_name] + r


def backup_grad(self, emb_name='word_embeddings'):
    for name, param in self.model.named_parameters():
        if param.requires_grad and emb_name in name:
            self.grad_backup[name] = param.grad.clone()


def restore_grad(self, emb_name='word_embeddings'):
    for name, param in self.model.named_parameters():
        if param.requires_grad and emb_name in name:
            param.grad = self.grad_backup[name]