import numpy as np
import torch
import torch.nn.functional as F


def LocalRecLoss(pred_embs, true_embs, pred_missing, true_missing, num_pred):
    CUDA, device = (pred_embs.device.type != 'cpu'), pred_embs.device
    if CUDA:
        true_missing = true_missing.cpu()
        pred_missing = pred_missing.cpu()
    pred_len = len(pred_embs)
    pred_embs = pred_embs.view(pred_len, num_pred, -1)
    loss = torch.zeros(pred_embs.shape[:2])
    if CUDA:
        loss = loss.to(device)

    pred_missing_np = np.round(
        pred_missing.detach().numpy()).reshape(-1).astype(np.int32)
    true_missing_np = true_missing.detach().numpy().reshape(-1).astype(
        np.int32)
    true_missing_np = np.clip(true_missing_np, 0, num_pred)
    pred_missing_np = np.clip(pred_missing_np, 0, num_pred)

    for i in range(pred_len):
        if true_missing_np[i] > 0:
            if isinstance(true_embs[i][true_missing_np[i] - 1], np.ndarray):
                true_emb_i = torch.tensor(true_embs[i]).to(device)
            else:
                true_emb_i = true_embs[i].to(device)
            for pred_j in range(min(num_pred, pred_missing_np[i])):
                true_embs_tensor = true_emb_i[true_missing_np[i] - 1]
                loss[i][pred_j] = F.mse_loss(
                    pred_embs[i][pred_j].unsqueeze(0).float(),
                    true_embs_tensor.unsqueeze(0).float())

                for true_k in range(min(num_pred, true_missing_np[i] - 1)):
                    true_embs_tensor = true_emb_i[true_k]

                    loss_ijk = F.mse_loss(
                        pred_embs[i][pred_j].unsqueeze(0).float(),
                        true_embs_tensor.unsqueeze(0).float())
                    if torch.sum(loss_ijk.data) < torch.sum(
                            loss[i][pred_j].data):
                        loss[i][pred_j] = loss_ijk

        else:
            continue
    # print(f"rec_loss是{loss.mean(1).mean(0).float()}")
    return loss.mean(1).mean(0).float()


def FedRecLoss(pred_embs, true_embs, pred_missing, num_pred):
    CUDA, device = (pred_embs.device.type != 'cpu'), pred_embs.device
    if CUDA:
        pred_missing = pred_missing.cpu()

    pred_len = len(pred_embs)
    pred_embs = pred_embs.view(pred_len, num_pred, -1)
    loss = torch.zeros(pred_embs.shape[:2])
    if CUDA:
        loss = loss.to(device)
    pred_missing_np = pred_missing.detach().numpy().reshape(-1).astype(
        np.int32)
    pred_missing_np = np.clip(pred_missing_np, 0, num_pred)
    if isinstance(true_embs[0], np.ndarray):
        true_embs = torch.tensor(true_embs).to(device)
    else:
        true_embs = true_embs.to(device)
    for i in range(pred_len):
        for pred_j in range(min(num_pred, pred_missing_np[i])):
            loss[i][pred_j] += F.mse_loss(
                pred_embs[i][pred_j].unsqueeze(0).float(),
                true_embs[i][0].unsqueeze(0).float())
            for true_k in true_embs[i][1:]:
                loss_ijk = F.mse_loss(
                    pred_embs[i][pred_j].unsqueeze(0).float(),
                    true_k.unsqueeze(0).float())
                if torch.sum(loss_ijk.data) < torch.sum(loss[i][pred_j].data):
                    loss[i][pred_j] = loss_ijk
    # print(f"rec_loss是{loss.mean(1).mean(0).float()}")
    return loss.mean(1).mean(0).float()
