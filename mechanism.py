import numpy as np
import torch
from torch.nn import functional as F


def weighted_decision_mechanism(num, s_list, t_list, alpha, beta, out_size=256):
    """
    num: The number of test samples
    s_list: List consisting of features outputted from different layers of student model
    t_list: List consisting of features outputted from different layers of teacher model
    alpha and beta: Hyperparameters for controlling upper and lower limit
    return: anomaly score for anomaly detection, anomaly map for anomaly segmentation
    """

    total_weights_list = list()
    n = len(s_list)
    output_list = [list() for _ in range(n)]
    for l, (t, s) in enumerate(zip(t_list, s_list)):
        output = 1 - F.cosine_similarity(t, s)  # minimize
        output_list[l].append(output)

    for i in range(num):
        low_similarity_list = list()
        for j in range(len(output_list)):
            low_similarity_list.append(torch.max(output_list[j][i]).cpu())
        probs = F.softmax(torch.tensor(low_similarity_list), 0)
        weight_list = list()  # set P consists of L high probability values, where L ranges from n-1 to n+1
        for idx, prob in enumerate(probs):
            weight_list.append(low_similarity_list[idx].numpy()) if prob > torch.mean(probs) else None
        weight = np.max([np.mean(weight_list) * alpha, beta])
        total_weights_list.append(weight)

    assert len(total_weights_list) == num, "the number of weights dose not match that of samples!"

    am_lists = [list() for _ in output_list]
    for l, output in enumerate(output_list):
        output = torch.cat(output, dim=0)
        a_map = torch.unsqueeze(output, dim=1)  # B*1*h*w
        am_lists[l] = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)[:, 0, :, :]  # B*256*256

    anomaly_map = sum(am_lists)
    anomaly_map_ = anomaly_map - anomaly_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]  # B*256*256
    anomaly_maps_exp = torch.exp(anomaly_map_)
    anomaly_score_exp = anomaly_maps_exp.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - anomaly_maps_exp

    batch = anomaly_score_exp.shape[0]
    anomaly_score = list()  # anomaly scores for all test samples
    for b in range(batch):
        top_k = int(out_size * out_size * total_weights_list[b])
        assert top_k >= 1 / (out_size * out_size), "weight can not be smaller than 1 / (H * W)!"

        single_anomaly_score_exp = anomaly_score_exp[b]
        assert single_anomaly_score_exp.reshape(1, -1).shape[-1] == out_size * out_size, \
            "something wrong with the last dimension of reshaped map!"

        single_map = single_anomaly_score_exp.reshape(1, -1)
        single_anomaly_score = np.mean(single_map.topk(top_k, dim=-1)[0].detach().cpu().numpy(), axis=1)
        anomaly_score.append(single_anomaly_score)

    return anomaly_score, anomaly_map.detach().cpu().numpy()


