import torch.nn.functional as F


def distillation_loss(s_logits, t_logits, labels, temperature, alpha):
    soft = F.kl_div(
        F.log_softmax(s_logits / temperature, dim=1),
        F.softmax(t_logits / temperature, dim=1),
        reduction='batchmean',
    ) * temperature ** 2
    hard = F.cross_entropy(s_logits, labels)
    return alpha * soft + (1 - alpha) * hard
