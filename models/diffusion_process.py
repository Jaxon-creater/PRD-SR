import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import enum
import math
class DiffusionProcess(nn.Module):
    def __init__(self,noise_schedule, noise_scale, noise_min, noise_max,steps, device, keep_num=10, personalized_scheduler=None):
        super(DiffusionProcess, self).__init__()
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device
        self.personalized_scheduler = personalized_scheduler  

        self.keep_num = keep_num
        self.Lt_record = torch.zeros(steps, keep_num, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)


        #The important parameters for gaussian diffusion
        self.beta_nums = torch.tensor(self.betas_num(), dtype=torch.float64).to(self.device)
        assert len(self.beta_nums.shape) == 1, "betas must be 1-D"
        assert len(self.beta_nums) == self.steps, "num of betas must equal to diffusion steps"
        assert (self.beta_nums > 0).all() and (self.beta_nums <= 1).all(), "betas out of range"

        self.diffusion_setting()

    def betas_num(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        st_bound = self.noise_scale * self.noise_min 
        e_bound = self.noise_scale * self.noise_max 
        if self.noise_schedule == "linear":
            return np.linspace(st_bound, e_bound, self.steps, dtype=np.float64)
        else:
            return betas_from_linear_variance(self.steps, np.linspace(st_bound, e_bound, self.steps, dtype=np.float64))
    def diffusion_setting(self):
        alphas = 1.0 - self.beta_nums 
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device) 
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(self.device)  
        assert self.alphas_cumprod_prev.shape == (self.steps,) 

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) 
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod) 
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod) 
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
            self.beta_nums * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) 
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.beta_nums * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
   

    def caculate_losses(self, model, emb_s, reweight=False, user_indices=None):
        """Compute denoising loss, optionally using personalized input noise."""
        batch_size, device = emb_s.size(0), emb_s.device 
        ts, pt = self.sample_timesteps(batch_size, device, 'uniform')
        noise = torch.randn_like(emb_s)
        
        
        noise_scale_tensor = self._personalized_scale(user_indices, device, batch_size)
        if self.personalized_scheduler is not None and user_indices is not None:
            noise = noise * noise_scale_tensor
        
        emb_t = self.forward_process(emb_s, ts, noise)
        terms = {}
        model_output = model(emb_t, ts, noise_scale_tensor)


        assert model_output.shape == emb_s.shape 

        mse = mean_flat((emb_s - model_output) ** 2)
        loss = mse

        if reweight == True:

            weight = self.SNR(ts - 1) - self.SNR(ts)
            weight = torch.where((ts == 0), 1.0, weight)
            loss = mse

        else:
            weight = torch.tensor([1.0] * len(model_output)).to(device)

        terms["loss"] = weight * loss
        terms["pred_xstart"] = model_output
        return terms

    def p_sample(self, model, emb_s, steps, sampling_noise=False, user_indices=None):
        """Run reverse diffusion from the requested starting timestep."""
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            return emb_s
        else:
            t = torch.tensor([steps - 1] * emb_s.shape[0]).to(emb_s.device)
            emb_t = self.q_sample(emb_s, t, user_indices=user_indices)

        noise_scale_tensor = self._personalized_scale(
            user_indices, emb_s.device, emb_s.shape[0]
        )
        indices = list(range(steps - 1, -1, -1))

        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * emb_t.shape[0]).to(emb_s.device)
                emb_t = model(emb_t, t, noise_scale_tensor)
            return emb_t

        for i in indices:
            t = torch.tensor([i] * emb_t.shape[0]).to(emb_s.device)
            out = self.p_mean_variance(model, emb_t, t, noise_scale_tensor)
            if sampling_noise:
                noise = torch.randn_like(emb_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(emb_t.shape) - 1)))
                )  # no noise when t == 0
                emb_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                emb_t = out["mean"]
        return emb_t

    def _personalized_scale(self, user_indices, device, batch_size):
        """Return xi_u for both SDNet conditioning and personalized variance."""
        if self.personalized_scheduler is None or user_indices is None:
            return torch.ones((batch_size, 1), dtype=torch.float32, device=device)
        return self.personalized_scheduler.get_noise_scale_tensor(user_indices, device)
    
 
       

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        """Sample diffusion timesteps uniformly or by historical loss."""
        if method == 'importance':  
            if not (self.Lt_count == self.keep_num).all():
                return self.sample_timesteps(batch_size, device, method='uniform')

            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_record ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt

        elif method == 'uniform':  # uniform sampling
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()

            return t, pt

        else:
            raise ValueError

    def forward_process(self, emb_s, t, noise=None):
        """Apply the closed-form forward diffusion process at timestep ``t``."""
        if noise is None:
            noise = torch.randn_like(emb_s)
        assert noise.shape == emb_s.shape
        return (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, emb_s.shape) * emb_s
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, emb_s.shape)
                * noise 
        )
    
    def q_sample(self, emb_s, t, noise=None, user_indices=None):
        """Sample the forward process with optional personalized noise scales."""
        if noise is None:
            noise = torch.randn_like(emb_s)
        
        
        if self.personalized_scheduler is not None and user_indices is not None:
            noise_scale_tensor = self.personalized_scheduler.get_noise_scale_tensor(user_indices, emb_s.device)
            noise = noise * noise_scale_tensor
        
        return self.forward_process(emb_s, t, noise)

    def q_posterior_mean_variance(self, emb_s, emb_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert emb_s.shape == emb_t.shape
        posterior_mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, emb_t.shape) * emb_s
                + self._extract_into_tensor(self.posterior_mean_coef2, t, emb_t.shape) * emb_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, emb_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, emb_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == emb_s.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, noise_scale=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, noise_scale)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        if noise_scale is not None:
            scale = noise_scale.to(device=x.device, dtype=x.dtype).view(-1, 1)
            model_variance = model_variance * scale.pow(2)
            model_log_variance = model_log_variance + 2.0 * torch.log(scale.clamp_min(1e-12))
        pred_xstart = model_output

        model_mean, _, _ = self.q_posterior_mean_variance(emb_s=pred_xstart, emb_t=x, t=t)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, emb_t, t, eps):
        assert emb_t.shape == eps.shape
        # Recover x_0 from x_t and the predicted noise epsilon.
        return (
                self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, emb_t.shape) * emb_t
                - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, emb_t.shape) * eps
        )

    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)



def betas_from_linear_variance(steps, variance, max_beta=0.999):
    """Recover per-step beta values from a cumulative variance schedule."""
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
