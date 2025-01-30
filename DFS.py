import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainRelated_Feature_Selection(nn.Module):
    def __init__(self, num_channels=256):
        super(DomainRelated_Feature_Selection, self).__init__()
        self.num_channels = num_channels    # the first layer
        #  initialize to 0
        self.theta1 = nn.Parameter(torch.zeros(1, num_channels, 1, 1)).to("cuda")
        self.theta2 = nn.Parameter(torch.zeros(1, num_channels * 2, 1, 1)).to("cuda")
        self.theta3 = nn.Parameter(torch.zeros(1, num_channels * 4, 1, 1)).to("cuda")

    def forward(self, xs, priors, conv=False):
        features = []
        for idx, (x, prior) in enumerate(zip(xs, priors)):
            #  to avoid losing local weight, theta should be as non-zero value as possible
            theta = torch.clamp(torch.sigmoid(eval("self.theta{}".format(idx+1))) * 1.0 + 0.5, max=1)

            b, c, h, w = x.shape
            if not conv:
                prior_flat = prior.view(b, c, -1)
                prior_flat_ = prior_flat.max(dim=-1, keepdim=True)[0]
                weights = F.softmax(prior_flat - prior_flat_, dim=-1)
                weights = weights.view(b, c, h, w)

                global_inf = prior.mean(dim=(-2, -1), keepdim=True)

                inter_weights = weights * (theta + global_inf)

            # alternative
            else:
                # ...
                # inter_weight = ...
                pass

            x_ = x * inter_weights
            features.append(x_)

        return features
