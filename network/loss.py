import torch
import numpy as np
import math
from configs import g_conf


def Action_nospeed_L1(params):
    print(params.keys())
    print(params['action_output'])
    B = params['action_output'].shape[0]  # batch_size

    # SingleFrame model - we only take into account the last frame's action
    mask_steer = (params['targets_action'][-1][:, 0] != -1000.0).detach()
    actions_loss_mat = torch.abs(params['action_output'][:, -1, :] - params['targets_action'][-1])  # (B, 2)

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


def apply_pose(points, yaw, x, y):
    """
    Apply differentiable SE(2) pose transformation (Yaw, X, Y) to 3D homogeneous points.

    Args:
        points: torch.Tensor of shape [1, 4] (homogeneous 3D point [x, y, z, 1])
        yaw: torch.Tensor of shape [batch_size, 1] (rotation around Z)
        x: torch.Tensor of shape [batch_size, 1] (translation along X)
        y: torch.Tensor of shape [batch_size, 1] (translation along Y)

    Returns:
        transformed_points: torch.Tensor of shape [batch_size, 4]
    """

    '''
    batch_size = yaw.shape[0]
    # return torch.stack([x, y, yaw, torch.ones_like(x)]).view(-1, batch_size).permute(1,0)

    # Compute sin and cos of yaw (keeps gradients)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # Rotation around Z axis (Yaw)
    rot = torch.zeros((batch_size, 4, 4), device=points.device, dtype=points.dtype)
    rot[:, 0, 0] = cos_yaw.squeeze()
    rot[:, 0, 1] = -sin_yaw.squeeze()
    rot[:, 1, 0] = sin_yaw.squeeze()
    rot[:, 1, 1] = cos_yaw.squeeze()
    rot[:, 2, 2] = 1.0
    rot[:, 3, 3] = 1.0

    # Translation
    rot[:, 0, 3] = x.squeeze()
    rot[:, 1, 3] = y.squeeze()

    # Apply transformation
    # points: [1, 4] -> expand to [batch_size, 4, 1] for batch matmul
    points_batched = points.expand(batch_size, -1).unsqueeze(-1)  # [batch_size, 4, 1]
    transformed_points = torch.bmm(rot, points_batched).squeeze(-1)  # [batch_size, 4]

    return transformed_points
    '''

    B = yaw.shape[0]

    cos_yaw = torch.cos(yaw)          # (B, 1)
    sin_yaw = torch.sin(yaw)          # (B, 1)

    # build 2x2 rotation blocks
    row0 = torch.cat([cos_yaw, -sin_yaw], dim=-1)  # (B, 2)
    row1 = torch.cat([sin_yaw,  cos_yaw], dim=-1)  # (B, 2)

    # full 4x4 matrices
    zeros = torch.zeros(B, 1, device=yaw.device, dtype=yaw.dtype)
    ones  = torch.ones(B, 1, device=yaw.device, dtype=yaw.dtype)

    # [ [R(0:2,0:2), t(0:2)],
    #   [ 0 0 1,     0     ],
    #   [ 0 0 0,     1     ] ]
    rot_row0 = torch.cat([row0, zeros, x], dim=-1)           # (B, 4)
    rot_row1 = torch.cat([row1, zeros, y], dim=-1)           # (B, 4)
    rot_row2 = torch.tensor([0., 0., 1., 0.], device=yaw.device, dtype=yaw.dtype).expand(B, 4)
    rot_row3 = torch.tensor([0., 0., 0., 1.], device=yaw.device, dtype=yaw.dtype).expand(B, 4)

    rot = torch.stack([rot_row0, rot_row1, rot_row2, rot_row3], dim=1)  # (B, 4, 4)

    pts = points.expand(B, -1).unsqueeze(-1)      # (B, 4, 1)
    transformed_points = torch.bmm(rot, pts).squeeze(-1)  # (B, 4)
    return transformed_points


def rollout_kinematic_model(v, wheelbase, steering_angle, acceleration, dt, steps):
    """
    Differentiable rollout of the kinematic bicycle model over multiple time steps.
    """

    B = steering_angle.shape[0]
    '''
    # clone state to avoid modifying inputs
    B = steering_angle.shape[0]
    x_t = 0
    y_t = 0
    yaw_t = torch.from_numpy(np.asarray([0.])).float().to('cuda')
    # with torch.no_grad():
    v_t = v # torch.from_numpy(np.asarray(v)).float().to('cuda')
    # points = torch.from_numpy(np.asarray([0., 0., 0., 1.])).float().to('cuda')
    points = torch.from_numpy(np.asarray([wheelbase, 0., 0., 1.])).float().to('cuda')
    '''

    x_t = torch.zeros(B, 1, device=steering_angle.device, dtype=steering_angle.dtype)
    y_t = torch.zeros_like(x_t)
    yaw_t = torch.zeros_like(x_t)
    v_t = v.view(B, 1)
    points = torch.tensor([0., 0., 0., 1.], device=steering_angle.device, dtype=steering_angle.dtype)

    transformed_points = []

    # current inputs
    steer_t = steering_angle.view(B, -1)  # [batch]
    accel_t = acceleration.view(B, -1)    # [batch]
    # print('steer_t: ', steer_t[0])
    max_steer = math.pi - 1e-4

    for t in range(steps):

        # kinematic bicycle model (differentiable)
        steer_safe = torch.clamp(steer_t, -max_steer, max_steer)
        yaw_dot = (v_t / wheelbase) * torch.tan(steer_safe)
        # print('yaw_dot: ', yaw_dot[0])
        x_dot = v_t * torch.cos(yaw_t)
        y_dot = v_t * torch.sin(yaw_t)

        # Euler integration
        x_t = x_t + x_dot * dt
        y_t = y_t + y_dot * dt
        yaw_t = yaw_t + yaw_dot * dt
        v_t = v_t + accel_t * dt

        # Apply transformation to point
        if g_conf.KBM_APPLY_POSE:
            p_trans = apply_pose(points, yaw_t, x_t, y_t)
        else:
            p_trans = torch.stack([x_t, y_t, yaw_t, torch.ones_like(x_t)]).view(-1, B).permute(1, 0)
        transformed_points.append(p_trans)

    transformed_points = torch.stack(transformed_points, dim=1)  # [batch, steps, 4]

    return transformed_points


def Action2WP_L1(params):
    B = params['action_output'].shape[0]  # batch_size

    mask_steer = params['targets_action'][-1][:, 0] != -1000.0
    actions_loss_mat = torch.abs(params['action_output'][:, -1, :] - params['targets_action'][-1])  # (B, 2)
    steer_loss = torch.sum(mask_steer * actions_loss_mat[:, 0]) / mask_steer.sum()
    acceleration_loss = torch.sum(actions_loss_mat[:, 1]) / B

    wheelbase = g_conf.KBM_WHEELBASE  # 1.45  # 1.7
    steps = g_conf.KBM_STEPS  #  8  # 20
    dt = g_conf.KBM_DT  #  0.5  # 0.1

    # speed denormalization
    speed_denorm = params['input_speed'][0] * (g_conf.DATA_NORMALIZATION['speed'][1] - g_conf.DATA_NORMALIZATION['speed'][0]) + g_conf.DATA_NORMALIZATION['speed'][0]

    # steering angle denormalization
    steer_pred = params['action_output'][:, -1, 0] * mask_steer * math.pi / 2.
    steer_gt = params['targets_action'][0][:, 0] * mask_steer * math.pi / 2.

    # acceleration denormalization
    acc_pred = params['action_output'][:, -1, 1] * 6.0
    acc_gt = params['targets_action'][0][:, 1] * 6.0

    weights_wp = torch.logspace(start=0, end=-(steps-1), steps=steps, base=2.0).view(1, -1, 1).float().cuda()
    # print('>>>> predictions')
    estimated_WP = rollout_kinematic_model(speed_denorm, wheelbase, steer_pred, acc_pred, dt, steps)
    # print('>>>> groundtruth')
    estimated_WP_gt = rollout_kinematic_model(speed_denorm, wheelbase, steer_gt, acc_gt, dt, steps)

    actions_loss_wp = torch.abs(estimated_WP - estimated_WP_gt) * weights_wp
    actions_loss_wp = torch.sum(actions_loss_wp) / B

    loss = actions_loss_wp * params['variable_weights']['actions']['wp'] + steer_loss * params['variable_weights']['actions']['steer'] + acceleration_loss * params['variable_weights']['actions']['acceleration']

    return loss, steer_loss, acceleration_loss, actions_loss_wp


def Loss(loss):
    if loss == 'Action_nospeed_L1':
        return Action_nospeed_L1
    elif loss == 'Action_nospeed_LN':
        return Action_nospeed_LN
    elif loss == 'Action_nospeed_SL':
        return Action_nospeed_SL
    elif loss == 'Action2WP_L1':
        return Action2WP_L1
    else:
        raise NotImplementError(" The loss of this model type has not yet defined ")
