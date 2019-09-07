import torch
import torch.nn as nn
import torch.nn.functional as F

class MattingLoss(nn.Module):
    def __init__(self):
        super(MattingLoss, self).__init__()

    def forward(self, img, gt_mask, alpha_matte, epsilon=1e-6):
        loss_alpha = self.f_loss(gt_mask, alpha_matte, epsilon)
        loss_color = self.f_loss(gt_mask.unsqueeze(1).repeat(1, 3, 1, 1).double() * img.double(),
                                alpha_matte.unsqueeze(1).repeat(1, 3, 1, 1).double() * img.double(), epsilon)

        # print(loss_alpha, loss_color)
        return loss_alpha + loss_color

    @staticmethod
    def f_loss(x1, x2, epsilon):
        # print(x1, x2)
        return torch.mean(torch.sqrt(torch.pow((x1.double() - x2.double()), 2)) + epsilon)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        device = target.device

        C = input.size(1)
        input = F.softmax(input, 1)
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, C) #N * C
        N = input.size(0)
        target = target.view(-1, 1) # N * 1
        target = torch.zeros(N, C).to(device).scatter_(1, target, torch.ones(N, 1).to(device)) # to one-hot, N*C

        if self.alpha is None:
            self.alpha = torch.ones(C, 1).to(device) / C

        loss = -torch.pow(1 - input, self.gamma) * input.log() * target #N*C

        loss = torch.matmul(loss, self.alpha)
        loss = loss.mean()

        return loss
