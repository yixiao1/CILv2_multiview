from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import g_conf


def Action_nospeed_L1(params: dict) -> Tuple[torch.Tensor, ...]:
    B = params['action_output'].shape[0]  # batch_size

    # SingleFrame model - we only take into account the last frame's action
    mask_steer = (params['targets_action'][-1][:, 0] != -1000.0).detach()
    actions_loss_mat = torch.abs(params['action_output'][:, -1, :] - params['targets_action'][-1])  # (B, 2)

    steer_loss = actions_loss_mat[:, 0] * params['variable_weights']['actions']['steer']
    num_valid_batch = mask_steer.sum().detach()
    steer_loss = torch.sum(steer_loss) / num_valid_batch

    if g_conf.ACCELERATION_AS_ACTION:
        acceleration_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['acceleration']
        acceleration_loss = torch.sum(acceleration_loss) / B

        loss = steer_loss + acceleration_loss

        return loss, steer_loss, acceleration_loss

    else:
        throttle_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['throttle']
        brake_loss = actions_loss_mat[:, 2] * params['variable_weights']['actions']['brake']
        throttle_loss = torch.sum(throttle_loss) / B
        brake_loss = torch.sum(brake_loss) / B

        loss = steer_loss + throttle_loss + brake_loss

        return loss, steer_loss, throttle_loss, brake_loss


def Action_nospeed_Quantile(params: dict) -> Tuple[torch.Tensor, ...]:
    """ Quantile regression loss """
    B = params['action_output'].shape[0]  # batch_size
    tau = 0.5 if 'tau' not in params['variable_weights'] else params['variable_weights']['tau']  # 0.5 * L1 loss by default

    # SingleFrame model - we only take into account the last frame's action
    mask_steer = (params['targets_action'][-1][:, 0] != -1000.0).detach()

    difference = params['action_output'][:, -1, :] - params['targets_action'][-1]  # (B, 2)
    actions_loss_mat = torch.zeros_like(difference, device=difference.device)  # (B, 2)

    # L1 loss on steering
    actions_loss_mat[:, 0] = torch.abs(difference[:, 0])

    # Quantile regression only on acceleration
    ind = (difference[:, 1] < 0.0).type(torch.float)
    actions_loss_mat[:, 1] = torch.abs((tau - ind) * difference[:, 1])

    # Weight each action differently
    steer_loss = actions_loss_mat[:, 0] * params['variable_weights']['actions']['steer']
    num_valid_batch = mask_steer.sum().detach()
    steer_loss = torch.sum(steer_loss) / num_valid_batch

    if g_conf.ACCELERATION_AS_ACTION:
        acceleration_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['acceleration']
        acceleration_loss = torch.sum(acceleration_loss) / B

        loss = steer_loss + acceleration_loss

        return loss, steer_loss, acceleration_loss

    else:
        throttle_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['throttle']
        brake_loss = actions_loss_mat[:, 2] * params['variable_weights']['actions']['brake']
        throttle_loss = torch.sum(throttle_loss) / B
        brake_loss = torch.sum(brake_loss) / B

        loss = steer_loss + throttle_loss + brake_loss

        return loss, steer_loss, throttle_loss, brake_loss


def Action_nospeed_LN(params):
    B = params['action_output'].shape[0]  # batch_size

    # SingleFrame model - we only take into account the last frame's action
    mask_steer = (params['targets_action'][-1][:, 0] != -1000.0).detach()
    actions_loss_mat = torch.abs(torch.pow(params['action_output'][:, -1, :] - params['targets_action'][-1], g_conf.LOSS_POW))  # (B, 2)

    steer_loss = mask_steer * actions_loss_mat[:, 0] * params['variable_weights']['actions']['steer']
    num_valid_batch = mask_steer.sum().detach()
    steer_loss = torch.sum(steer_loss) / num_valid_batch

    if g_conf.ACCELERATION_AS_ACTION:
        acceleration_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['acceleration']
        acceleration_loss = torch.sum(acceleration_loss) / B

        loss = steer_loss + acceleration_loss

        return loss, steer_loss, acceleration_loss

    else:
        throttle_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['throttle']
        brake_loss = actions_loss_mat[:, 2] * params['variable_weights']['actions']['brake']
        throttle_loss = torch.sum(throttle_loss) / B
        brake_loss = torch.sum(brake_loss) / B

        loss = steer_loss + throttle_loss + brake_loss

        return loss, steer_loss, throttle_loss, brake_loss


def Action_nospeed_SL(params):
    B = params['action_output'].shape[0]  # batch_size

    # SingleFrame model - we only take into account the last frame's action
    a = 10.0
    c = 0.2
    mask_steer = (params['targets_action'][-1][:, 0] != -1000.0).detach()

    actions_loss_mat_l1 = torch.abs(params['action_output'][:, -1, :] - params['targets_action'][-1])  # (B, 2)
    actions_loss_mat_l2 = torch.pow(actions_loss_mat_l1, 2)  # (B, 2)
    actions_loss_mat = actions_loss_mat_l2 / (1.0 + torch.exp(a * (c - actions_loss_mat_l1)))

    steer_loss = mask_steer * actions_loss_mat[:, 0] * params['variable_weights']['actions']['steer']
    num_valid_batch = mask_steer.sum().detach()
    steer_loss = torch.sum(steer_loss) / num_valid_batch

    if g_conf.ACCELERATION_AS_ACTION:
        acceleration_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['acceleration']
        acceleration_loss = torch.sum(acceleration_loss) / B

        loss = steer_loss + acceleration_loss

        return loss, steer_loss, acceleration_loss

    else:
        throttle_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['throttle']
        brake_loss = actions_loss_mat[:, 2] * params['variable_weights']['actions']['brake']
        throttle_loss = torch.sum(throttle_loss) / B
        brake_loss = torch.sum(brake_loss) / B

        loss = steer_loss + throttle_loss + brake_loss

        return loss, steer_loss, throttle_loss, brake_loss


def Action_nospeed_L1_Attention_KL(params):
    B = params['action_output'].shape[0]  # batch_size

    # SingleFrame model - we only take into account the last frame's action
    mask_steer = (params['targets_action'][-1][:, 0] != -1000.0).detach()
    actions_loss_mat = torch.abs(params['action_output'][:, -1, :] - params['targets_action'][-1])  # (B, 2)

    steer_loss = actions_loss_mat[:, 0] * params['variable_weights']['actions']['steer']
    num_valid_batch = mask_steer.sum().detach()
    steer_loss = torch.sum(steer_loss) / num_valid_batch

    # Attention loss
    eps = 1e-12  # For numerical stability
    att_loss = params['variable_weights']['attention'] * F.kl_div((params['attention_output']+eps).log(),  # Transf. Encoder attention map (GAPn)
                                                                  params['targets_attention'], # Ground truth attention map (virtual or human)
                                                                  reduction='batchmean')

    if g_conf.ACCELERATION_AS_ACTION:
        acceleration_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['acceleration']
        acceleration_loss = torch.sum(acceleration_loss) / B

        loss = steer_loss + acceleration_loss + att_loss

        return loss, steer_loss, acceleration_loss, att_loss

    else:
        throttle_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['throttle']
        brake_loss = actions_loss_mat[:, 2] * params['variable_weights']['actions']['brake']
        throttle_loss = torch.sum(throttle_loss) / B
        brake_loss = torch.sum(brake_loss) / B

        loss = steer_loss + throttle_loss + brake_loss + att_loss

        return loss, steer_loss, throttle_loss, brake_loss, att_loss



def Action_nospeed_Quantile_Attention_KL(params):
    B = params['action_output'].shape[0]  # batch_size
    tau = 0.5 if 'tau' not in params['variable_weights'] else params['variable_weights']['tau']  # 0.5 * L1 loss by default

    # SingleFrame model - we only take into account the last frame's action
    mask_steer = (params['targets_action'][-1][:, 0] != -1000.0).detach()

    difference = params['action_output'][:, -1, :] - params['targets_action'][-1]  # (B, 2)
    actions_loss_mat = torch.zeros_like(difference, device=difference.device)  # (B, 2)

    # L1 loss on steering
    actions_loss_mat[:, 0] = torch.abs(difference[:, 0])

    # Quantile regression only on acceleration
    ind = (difference[:, 1] < 0.0).type(torch.float)
    actions_loss_mat[:, 1] = torch.abs((tau - ind) * difference[:, 1])

    # Weight each action differently
    steer_loss = actions_loss_mat[:, 0] * params['variable_weights']['actions']['steer']
    num_valid_batch = mask_steer.sum().detach()
    steer_loss = torch.sum(steer_loss) / num_valid_batch


    # Attention loss
    eps = 1e-12  # For numerical stability
    att_loss = params['variable_weights']['attention'] * F.kl_div((params['attention_output']+eps).log(),  # Transf. Encoder attention map (GAPn)
                                                                  params['targets_attention'], # Ground truth attention map (virtual or human)
                                                                  reduction='batchmean')

    if g_conf.ACCELERATION_AS_ACTION:
        acceleration_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['acceleration']
        acceleration_loss = torch.sum(acceleration_loss) / B

        loss = steer_loss + acceleration_loss + att_loss

        return loss, steer_loss, acceleration_loss, att_loss

    else:
        throttle_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['throttle']
        brake_loss = actions_loss_mat[:, 2] * params['variable_weights']['actions']['brake']
        throttle_loss = torch.sum(throttle_loss) / B
        brake_loss = torch.sum(brake_loss) / B

        loss = steer_loss + throttle_loss + brake_loss + att_loss

        return loss, steer_loss, throttle_loss, brake_loss, att_loss


def Action_nospeed_L1_Attention_L2(params):
    B = params['action_output'].shape[0]  # batch_size

    # SingleFrame model - we only take into account the last frame's action
    mask_steer = (params['targets_action'][-1][:, 0] != -1000.0).detach()
    actions_loss_mat = torch.abs(params['action_output'][:, -1, :] - params['targets_action'][-1])  # (B, 2)

    steer_loss = actions_loss_mat[:, 0] * params['variable_weights']['actions']['steer']
    num_valid_batch = mask_steer.sum().detach()
    steer_loss = torch.sum(steer_loss) / num_valid_batch

    # Attention loss
    att_loss = params['variable_weights']['attention'] * F.mse_loss(params['attention_output'], params['targets_attention'])

    if g_conf.ACCELERATION_AS_ACTION:
        acceleration_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['acceleration']
        acceleration_loss = torch.sum(acceleration_loss) / B

        loss = steer_loss + acceleration_loss + att_loss

        return loss, steer_loss, acceleration_loss, att_loss

    else:
        throttle_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['throttle']
        brake_loss = actions_loss_mat[:, 2] * params['variable_weights']['actions']['brake']
        throttle_loss = torch.sum(throttle_loss) / B
        brake_loss = torch.sum(brake_loss) / B

        loss = steer_loss + throttle_loss + brake_loss + att_loss

        return loss, steer_loss, throttle_loss, brake_loss, att_loss

def Action_nospeed_L1_mhaAttention_Cossim(params):
    B = params['action_output'].shape[0]  # batch_size

    # SingleFrame model - we only take into account the last frame's action
    mask_steer = (params['targets_action'][-1][:, 0] != -1000.0).detach()
    actions_loss_mat = torch.abs(params['action_output'][:, -1, :] - params['targets_action'][-1])  # (B, 2)

    steer_loss = actions_loss_mat[:, 0] * params['variable_weights']['actions']['steer']
    num_valid_batch = mask_steer.sum().detach()
    steer_loss = torch.sum(steer_loss) / num_valid_batch

    # Attention loss
    # Flatten each head's attention map
    flattened_maps = torch.mean(params['mha_attention_output'], dim=2)  # (B, h, N, N) -> (B, h, N)
    num_heads = flattened_maps.shape[1]

    expanded_maps = flattened_maps.unsqueeze(1)  # Shape [B, 1, h, N]
    transposed_maps = flattened_maps.unsqueeze(2)  # Shape [B, h, 1, N]

    # Cosine similarity module
    cossim = nn.CosineSimilarity(dim=3)

    # Calculate all pairwise cosine similarities at once
    pairwise_similarities = cossim(expanded_maps, transposed_maps)

    # Create a mask to zero out self-comparisons and lower triangle
    mask = torch.triu(torch.ones(num_heads, num_heads, device=flattened_maps.device), diagonal=1)

    # Apply the mask and sum the valid similarities
    valid_similarities = (1 - pairwise_similarities) * mask

    # Compute the loss, normalize by batch size, apply weighting
    mha_att_cossim_loss = params['variable_weights']['mha_cossim'] * valid_similarities.sum() / B

    class HLoss(nn.Module):
        def __init__(self):
            super(HLoss, self).__init__()

        def forward(self, x):
            b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
            b = -1.0 * b.sum()
            return b

    # entropy_regularization_criterion = HLoss()
    if params['entropy_reg']:
        entropy_regularization = torch.distributions.Categorical(
            probs=flattened_maps.mean(dim=1)).entropy().sum()
        mha_att_cossim_loss += params['variable_weights']['mha_entro_reg'] * entropy_regularization

    # total_cossim = int(num_heads * (num_heads - 1) // 2)  # Total number of cosine similarities to calculate; triangular number
    # cossim = nn.CosineSimilarity(dim=1)

    # # Compute pairwise cosine similarity
    # cosine_similarities = torch.zeros(total_cossim, device=flattened_maps.device)
    # idx = 0
    # for i in range(num_heads):
    #     for j in range(i+1, num_heads):  # Cosine similarity is symmetric
    #         cosine_similarities[idx] = (1-cossim(flattened_maps[:, i], flattened_maps[:, j])).sum()
    #         idx += 1
    # mha_att_cossim_loss = params['variable_weights']['mha_cossim'] * cosine_similarities.detach().sum() / B
    # print(mha_att_cossim_loss)

    if g_conf.ACCELERATION_AS_ACTION:
        acceleration_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['acceleration']
        acceleration_loss = torch.sum(acceleration_loss) / B

        loss = steer_loss + acceleration_loss + mha_att_cossim_loss

        return loss, steer_loss, acceleration_loss, mha_att_cossim_loss

    else:
        throttle_loss = actions_loss_mat[:, 1] * params['variable_weights']['actions']['throttle']
        brake_loss = actions_loss_mat[:, 2] * params['variable_weights']['actions']['brake']
        throttle_loss = torch.sum(throttle_loss) / B
        brake_loss = torch.sum(brake_loss) / B

        loss = steer_loss + throttle_loss + brake_loss + mha_att_cossim_loss

        return loss, steer_loss, throttle_loss, brake_loss, mha_att_cossim_loss


def Loss(loss):
    if loss == 'Action_nospeed_L1':
        return Action_nospeed_L1
    elif loss == 'Action_nospeed_LN':
        return Action_nospeed_LN
    elif loss == 'Action_nospeed_SL':
        return Action_nospeed_SL
    elif loss == 'Action_nospeed_Quantile':
        return Action_nospeed_Quantile
    elif loss == 'Action_nospeed_L1_Attention_KL':
        return Action_nospeed_L1_Attention_KL
    elif loss == 'Action_nospeed_Quantile_Attention_KL':
        return Action_nospeed_Quantile_Attention_KL
    elif loss == 'Action_nospeed_L1_Attention_L2':
        return Action_nospeed_L1_Attention_L2
    elif loss == 'Action_nospeed_L1_mhaAttention_Cossim':
        return Action_nospeed_L1_mhaAttention_Cossim
    else:
        raise NotImplementedError("The loss of this model type has not yet defined ")
