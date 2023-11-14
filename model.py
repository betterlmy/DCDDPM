from DCUnet import *
import torch.nn.functional as F


class DCDDPM(nn.Module):
    def __init__(self, unet_model, beta_schedule="linear", T=1000):
        super(DCDDPM, self).__init__()
        self.seed = 1
        self.unet = unet_model
        self.T = T  # Number of diffusion steps
        self.beta_schedule = self._get_beta_schedule(beta_schedule, T)

    def _get_beta_schedule(self, schedule, T):
        if schedule == "linear":
            return torch.linspace(0.0001, 0.02, T).to(self.unet.device)
        else:
            raise ValueError("Unknown beta schedule")

    def control_seed(self):
        torch.manual_seed(self.seed)
        self.seed += 1

    def add_noise_sample(self, x_start, t, noise=None):
        """ Sample from q(x_t | x_0) """
        self.control_seed()
        if noise is None:
            noise = torch.randn_like(x_start)
        beta_t = self.beta_schedule[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return torch.sqrt(1 - beta_t) * x_start + torch.sqrt(beta_t) * noise, torch.sqrt(beta_t) * noise

    # def p_sample(self, x_t, t):
    #     """ Sample from p(x_{t-1} | x_t) """
    #     if torch.all(t) == 0:
    #         return x_t
    #     else:
    #         beta_t = self.beta_schedule[t]
    #         pred_noise = self.unet(x_t, t)
    #         mean = (1 / torch.sqrt(1 - beta_t)) * (x_t - (beta_t / torch.sqrt(1 - beta_t)) * pred_noise)
    #         noise = torch.randn_like(x_t)
    #         return mean + torch.sqrt(beta_t) * noise

    def get_loss(self, x_t, scaled_noise, t):
        """ Get loss for training """
        self.control_seed()

        pred_noise = self.unet(x_t)  # self.unet(x_t, t)
        loss = F.mse_loss(pred_noise, scaled_noise, reduction="sum")
        # loss = torch.mean((x_recon - x_start) ** 2)
        return loss/x_t.size(0)

    def forward(self, x_start):
        self.control_seed()
        t = torch.randint(0, self.T, (x_start.size(0),), device=self.unet.device)
        x_t, scaled_noise = self.add_noise_sample(x_start, t)
        return self.get_loss(x_t, scaled_noise, t)
