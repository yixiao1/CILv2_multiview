import torch
import numpy as np
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
    batch_size = yaw.shape[0]

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


def rollout_kinematic_model(v, wheelbase, steering_angle, acceleration, dt, steps):
    """
    Differentiable rollout of the kinematic bicycle model over multiple time steps.
    """
    # clone state to avoid modifying inputs
    B = steering_angle.shape[0]
    x_t = 0
    y_t = 0
    yaw_t = torch.from_numpy(np.asarray([0.])).float().to('cuda')
    # with torch.no_grad():
    v_t = v # torch.from_numpy(np.asarray(v)).float().to('cuda')
    # points = torch.from_numpy(np.asarray([0., 0., 0., 1.])).float().to('cuda')
    points = torch.from_numpy(np.asarray([g_conf.KBM_WHEELBASE, 0., 0., 1.])).float().to('cuda')

    transformed_points = []

    # current inputs
    steer_t = steering_angle.view(B, -1)  # [batch]
    accel_t = acceleration.view(B, -1)    # [batch]

    for t in range(steps):

        # kinematic bicycle model (differentiable)
        yaw_dot = (v_t / wheelbase) * torch.tan(steer_t)
        x_dot = v_t * torch.cos(yaw_t)
        y_dot = v_t * torch.sin(yaw_t)

        # Euler integration
        x_t = x_t + x_dot * dt
        y_t = y_t + y_dot * dt
        yaw_t = yaw_t + yaw_dot * dt
        v_t = v_t + accel_t * dt

        # Apply transformation to point
        p_trans = apply_pose(points, yaw_t, x_t, y_t)
        transformed_points.append(p_trans)

    transformed_points = torch.stack(transformed_points, dim=1)  # [batch, steps, 4]

    return transformed_points


def Action2WP_L1(params):
    B = params['action_output'].shape[0]  # batch_size

    actions_loss_mat = torch.abs(params['action_output'][:, -1, :] - params['targets_action'][-1])  # (B, 2)
    steer_loss = torch.sum(actions_loss_mat[:, 0]) / B
    acceleration_loss = torch.sum(actions_loss_mat[:, 1]) / B

    wheelbase = g_conf.KBM_WHEELBASE  # 1.45  # 1.7
    steps = g_conf.KBM_STEPS  #  8  # 20
    dt = g_conf.KBM_DT  #  0.5  # 0.1

    # speed denormalization
    # params['input_speed'] = np.asarray(params['input_speed'])
    params['input_speed'][0] = params['input_speed'][0] * (g_conf.DATA_NORMALIZATION['speed'][1] - g_conf.DATA_NORMALIZATION['speed'][0]) + g_conf.DATA_NORMALIZATION['speed'][0]

    # steering angle denormalization
    params['action_output'][:, -1, 0] = params['action_output'][:, -1, 0] * 3.14159265359
    params['targets_action'][0][:, 0] = params['targets_action'][0][:, 0] * 3.14159265359

    # acceleration denormalization
    params['action_output'][:, -1, 1] = params['action_output'][:, -1, 1] * 6.0
    params['targets_action'][0][:, 1] = params['targets_action'][0][:, 1] * 6.0

    estimated_WP = rollout_kinematic_model(params['input_speed'][0], wheelbase, params['action_output'][:, -1, 0], params['action_output'][:, -1, 1], dt, steps)
    estimated_WP_gt = rollout_kinematic_model(params['input_speed'][0], wheelbase, params['targets_action'][0][:, 0], params['targets_action'][0][:, 1], dt, steps)

    actions_loss_wp = torch.abs(estimated_WP - estimated_WP_gt)
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
