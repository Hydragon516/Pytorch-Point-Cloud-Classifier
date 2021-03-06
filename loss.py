import torch
import torch.nn as nn
import torch.nn.functional as F

class loss_for_pointnet(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(loss_for_pointnet, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
    
    def feature_transform_reguliarzer(self, trans):
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
        
        return loss

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = self.feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale

        return total_loss


def loss_for_dgcnn(pred, gold, smoothing=True):

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss