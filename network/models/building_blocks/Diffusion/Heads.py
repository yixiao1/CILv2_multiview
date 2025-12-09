import torch
import torch.nn as nn

from typing import Optional, Dict

from diffusers.schedulers import DDIMScheduler


# ------------------------------------------------------------------
# DP-style Action Diffusion Head
#   - No anchors, no modes, continuous (throttle, steer, brake)
#   - Training: epsilon prediction (MSE)
#   - Inference: DDIM sampling
#   - Inspired in diffusion_transformer_hybrid_image_policy (https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py)
# ------------------------------------------------------------------

class ActionDiffusionDPHead(nn.Module):
    """
    Wraps a diffusion transformer with a DDPM scheduler.

    Args:
        model: a callable module(sample, timestep, cond) -> (B, T, input_dim)
               (we pass a TransformerCILppForDiffusion wired to the CIL++ TD)
        input_dim: action token dim (3 for [throttle, steer, brake])
        horizon: number of action tokens (T)
        num_train_timesteps: scheduler training steps
        prediction_type: 'epsilon'
        num_inference_steps: sampling steps at inference
    """
    def __init__(self,
                 model: nn.Module,
                 input_dim: int = 3,
                 horizon: int = 1,
                 num_train_timesteps: int = 1000,
                 beta_schedule: str = "scaled_linear",
                 prediction_type: str = 'epsilon',
                 num_inference_steps: int = 2
                 ):
        super().__init__()
        self.model = model
        self.input_dim = input_dim
        self.horizon = horizon

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type
        )
        self.prediction_type = prediction_type
        self.num_inference_steps = num_inference_steps

        self.loss_fn = nn.MSELoss(reduction='mean')

    # -------------------------------
    # Unified forward (train/eval)
    # -------------------------------
    def forward(self,
                cond: torch.tensor,
                targets: Optional[Dict[str, torch.tensor]] = None,
                num_steps: Optional[int] = None) -> Dict[str, torch.tensor]:
        if self.training:
            return self.forward_train(cond=cond, targets=targets)
        else:
            return self.forward_test(cond=cond, num_steps=num_steps)

    # -------------------------------
    # Training step
    # -------------------------------
    def forward_train(self,
                      cond: torch.Tensor,  # (B, To, cond_dim)
                      targets: torch.Tensor  # expects targets['actions'] = (B, T, 3)
                      ) -> Dict[str, torch.Tensor]:
        """
        DP training (epsilon prediction):
          - Sample timestep k
          - Add noise to clean actions
          - Predict epsilon with the transformer
          - MSE(pred_eps, true_eps)
        """
        clean = targets  # (B, T, 3)
        B, T, D = clean.shape
        assert T == self.horizon and D == self.input_dim, f"Expected actions (B,{self.horizon},{self.input_dim}), got {clean.shape}"

        device = clean.device
        # sample timesteps per batch element
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(B,),
            device=device,
            dtype=torch.long
        )

        noise = torch.randn_like(clean)
        noisy = self.noise_scheduler.add_noise(clean, noise, timesteps)  # (B, T, 3)

        # predict epsilon
        pred = self.model(sample=noisy, timestep=timesteps, cond=cond)  # (B,T,3)

        # epsilon-prediction loss
        loss = self.loss_fn(pred, noise)

        return {
            "action_loss": loss,
            "denoise_pred": pred,  # optional for debugging
            "noisy_actions": noisy,    # optional
            "timesteps": timesteps
        }

    # -------------------------------
    # Inference (sampling)
    # -------------------------------
    @torch.no_grad()
    def forward_test(self,
                     cond: torch.Tensor,    # (B, To, cond_dim)
                     num_steps: Optional[int] = None
                     ) -> Dict[str, torch.Tensor]:
        """
        DDIM sampling:
          - Start from Gaussian noise
          - Iterate scheduler timesteps
          - At each step, model predicts epsilon; scheduler steps sample
        """
        # B = cond.shape[0] #cond.shape
        # device = cond.device
        # dtype = cond.dtype

        # # initialize noise
        # sample = torch.randn((B, self.horizon, self.input_dim), device=device, dtype=dtype)

        B = int(cond.shape[0])
        H = int(self.horizon)
        D = int(self.input_dim)
        sample = torch.empty(B, H, D, device=cond.device, dtype=cond.dtype).normal_()

        # set inference timesteps (DDIM)
        steps = self.num_inference_steps if num_steps is None else int(num_steps)
        self.noise_scheduler.set_timesteps(num_inference_steps=steps, device=cond.device)

        for t in self.noise_scheduler.timesteps:
            # model predicts epsilon (or v/x0 depending on prediction_type)
            model_out = self.model(sample=sample, timestep=t, cond=cond)
            step = self.noise_scheduler.step(model_output=model_out, timestep=t, sample=sample)
            sample = step.prev_sample  # DDIM next latent

        # 'sample' is the final actions after denoising
        return {
            "action_pred": sample  # (B, T, 3)
        }
