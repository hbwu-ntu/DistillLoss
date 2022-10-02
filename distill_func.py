"""
Reference: https://github.com/HobbitLong/RepDistiller
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Utterance-level:
1. Correlation_loss
"""


class Correlation_loss(nn.Module):
    """
    Correlation Congruence for Knowledge Distillation
    https://arxiv.org/pdf/1904.01802.pdf
    """

    def __init__(self):
        super(Correlation_loss, self).__init__()
        self.layer = nn.Linear(2, 2)  # NOTE: a trick; won't train it

    def forward(self, teacher_layers, student_layers, pred, target):
        """
        Args:
            teacher_layers (List[str]): indicating the layers to calculate loss
            student_layers (List[str]): indicating the layers to calculate loss
            NOTE: the len(teacher_layers) can be different from len(student_layers)
            pred (dict(torch.FloatTensor)): prediction from student (B, T, hidden_size)
            target (dict(torch.FloatTensor)): target from teacher (B, T, hidden_size)
        Returns:
            loss (torch.float)
        """
        # NOTE: take the last layer embeddings of the teacher and student
        f_s = pred[student_layers[-1]]
        f_t = target[teacher_layers[-1]]

        f_s = torch.mean(f_s, dim=1)
        f_t = torch.mean(f_t, dim=1)

        loss = self.similarity_loss(f_s, f_t)
        return loss

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = G_s / G_s.norm(2)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = G_t / G_t.norm(2)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss


def test_Correlation():
    pred = {}
    pred["layer_1"] = torch.rand(4, 100, 256)
    target = {}
    target["layer_1"] = torch.rand(4, 100, 256)
    teacher_layers = ["layer_1"]
    student_layers = ["layer_1"]
    embd_loss = Correlation_loss()
    loss = embd_loss(teacher_layers, student_layers, pred, target)
    print(loss, type(loss))


"""
Embedding-level:
1. Multitask embedding loss
2. Layerwise embedding loss
"""


class Multitask_embedding_loss(nn.Module):
    """
    Multitask embedding loss
    paper: DistillHubert https://arxiv.org/abs/2110.01900
    """

    def __init__(self, student_dim, teacher_dim, loss_type="l1", cos_weight=1.0):
        """
        Args:
            student_dim (List[float]): layerwise embedding dim of student
            teacher_dim (List[float]): layerwise embedding dim of teacher
            loss_type (str): whether l1 or l2 loss
        """
        super(Multitask_embedding_loss, self).__init__()

        if loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="none")
        elif loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(loss_type)

        self.cos_weight = cos_weight

        self.layers = nn.ModuleList([])
        for idx in range(len(student_dim)):
            new_layer = nn.Linear(student_dim[idx], teacher_dim[idx])
            self.layers.append(new_layer)

    def forward(self, teacher_layers, student_layers, pred, target):
        """
        Args:
            teacher_layers (List[str]): indicating the layers to calculate loss
            student_layers (List[str]): indicating the layers to calculate loss
            NOTE: the len(teacher_layers) can be different from len(student_layers)
            pred (dict(torch.FloatTensor)): prediction from student (B, T, hidden_size)
            target (dict(torch.FloatTensor)): target from teacher (B, T, hidden_size)
        Returns:
            total_loss (Torch.float)
        """
        total_loss = 0

        for idx in range(len(teacher_layers)):
            # NOTE: DistillHubert uses the last layer to predict the hidden embeddings of each teacher layer
            stud_pred = self.layers[idx](pred[student_layers[-1]])
            sim_loss = -F.logsigmoid(
                F.cosine_similarity(stud_pred, target[teacher_layers[idx]], dim=-1)
            )
            sim_loss = sim_loss.mean()
            rec_loss = self.loss_func(stud_pred, target[teacher_layers[idx]])
            rec_loss = rec_loss.mean()
            total_loss = rec_loss + self.cos_weight * sim_loss

        return total_loss


def test_Multitask_embedding_loss():
    pred = {}
    pred["layer_1"] = torch.rand(4, 100, 256)
    target = {}
    target["layer_1"] = torch.rand(4, 100, 256)
    teacher_layers = ["layer_1"]
    student_layers = ["layer_1"]
    student_dim = [256]
    teacher_dim = [256]
    embd_loss = Multitask_embedding_loss(student_dim, teacher_dim)
    loss = embd_loss(teacher_layers, student_layers, pred, target)
    print(loss)


class Layerwise_embedding_loss(Multitask_embedding_loss):
    """
    Layerwise embedding loss
    paper: Deep versus Wide https://arxiv.org/pdf/2207.06867.pdf
    NOTE: this is for embedding dimentions are different for student and teacher
    """

    def __init__(self, student_dim, teacher_dim, loss_type="l1", cos_weight=1.0):
        super().__init__(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            loss_type=loss_type,
            cos_weight=cos_weight,
        )

    def forward(self, teacher_layers, student_layers, pred, target):
        """
        Args:
            teacher_layers (List[str]): indicating the layers to calculate loss
            student_layers (List[str]): indicating the layers to calculate loss
            NOTE: the len(teacher_layers) can be different from len(student_layers)
            pred (dict(torch.FloatTensor)): prediction from student (B, T, hidden_size)
            target (dict(torch.FloatTensor)): target from teacher (B, T, hidden_size)
        Returns:
            total_loss (Torch.float)
        """
        total_loss = 0

        for idx in range(len(student_layers)):
            stud_pred = self.layers[idx](pred[student_layers[idx]])
            sim_loss = -F.logsigmoid(
                F.cosine_similarity(stud_pred, target[teacher_layers[idx]], dim=-1)
            )
            sim_loss = sim_loss.mean()
            rec_loss = self.loss_func(stud_pred, target[teacher_layers[idx]])
            rec_loss = rec_loss.mean()
            total_loss = rec_loss + self.cos_weight * sim_loss

        return total_loss


def test_Layerwise_embedding_loss():
    pred = {}
    pred["layer_1"] = torch.rand(4, 100, 256)
    target = {}
    target["layer_1"] = torch.rand(4, 100, 256)
    teacher_layers = ["layer_1"]
    student_layers = ["layer_1"]
    student_dim = [256]
    teacher_dim = [256]
    embd_loss = Layerwise_embedding_loss(student_dim, teacher_dim)
    loss = embd_loss(teacher_layers, student_layers, pred, target)
    print(loss)


if __name__ == "__main__":
    # test_Correlation()
    test_Layerwise_embedding_loss()
    test_Multitask_embedding_loss()
