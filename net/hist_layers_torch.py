import numpy as np
import torch
import torch.nn.functional as F

class HistogramLayers(object):
    """Network augmentation for 1D and 2D (Joint) histograms construction,
    Calculate Earth Mover's Distance, Mutual Information loss
    between output and target
    """

    def __init__(self, out, tar):
        self.bin_num = 256
        self.min_val = 0.0
        self.max_val = 1.0
        self.interval_length = (self.max_val - self.min_val) / self.bin_num
        self.kernel_width = self.interval_length / 2.5
        self.maps_out = self.calc_activation_maps(out)  # (256)
        self.bs, self.n_pixels, _ = self.maps_out.shape
        t = tar
        for i in range(1, self.bs):
            t = torch.cat((t, tar), 0)
        self.maps_tar_p = t  # (256)

    def calc_activation_maps(self, img):
        # apply approximated shifted rect (bin_num) functions on img
        bins_min_max = np.linspace(self.min_val, self.max_val, self.bin_num + 1)
        bins_av = (bins_min_max[0:-1] + bins_min_max[1:]) / 2
        bins_av = torch.tensor(bins_av, dtype=torch.float32).cuda()  # shape = (,bin_num)
        bins_av = torch.unsqueeze(bins_av, dim=0)  # shape = (1,bin_num)
        bins_av = torch.unsqueeze(bins_av, dim=0)  # shape = (1,1,bin_num)
        img_flat = torch.unsqueeze(torch.flatten(img, start_dim=1), dim=-1)  # (batch_size,H*W,1)
        maps = self.activation_func(img_flat, bins_av)  # shape = (batch_size,H*W,bin_num)
        return maps

    def activation_func(self, img_flat, bins_av):
        img_minus_bins_av = torch.sub(img_flat, bins_av)  # shape=  (batch_size,H*W,bin_num)
        img_plus_bins_av = torch.add(img_flat, bins_av)  # shape = (batch_size,H*W,bin_num)
        sigmoid = torch.nn.Sigmoid()
        maps = sigmoid((img_minus_bins_av + self.interval_length / 2) / self.kernel_width) \
               - sigmoid((img_minus_bins_av - self.interval_length / 2) / self.kernel_width) \
               + sigmoid((img_plus_bins_av - 2 * self.min_val + self.interval_length / 2) / self.kernel_width) \
               - sigmoid((img_plus_bins_av - 2 * self.min_val - self.interval_length / 2) / self.kernel_width) \
               + sigmoid((img_plus_bins_av - 2 * self.max_val + self.interval_length / 2) / self.kernel_width) \
               - sigmoid((img_plus_bins_av - 2 * self.max_val - self.interval_length / 2) / self.kernel_width)
        return maps

    def ecdf(self, maps):
        # calculate the CDF of p
        p = torch.sum(maps, 1) / self.n_pixels  # shape=(batch_size,bin_bum)
        return torch.cumsum(p, 1)  # 累计概率

    def emd_loss(self, maps_p_t, maps_o):
        ecdf_p = torch.cumsum(maps_p_t, 1)  # shape=(batch_size, bin_bum)
        ecdf_p_hat = self.ecdf(maps_o)  # shape=(batch_size, bin_bum)
        emd = torch.mean(torch.pow(torch.abs(ecdf_p - ecdf_p_hat), 2), dim=-1)  # shape=(batch_size,1)
        emd = torch.pow(emd, 1 / 2)
        return torch.mean(emd)  # shape=0
    def p(self, maps):
        # calculate p
        p = torch.sum(maps, 1) / self.n_pixels  # shape=(batch_size,bin_bum)
        return p
    def kl_loss(self, maps_p_t, maps_o):
        p = maps_p_t  # shape=(batch_size, bin_bum)
        p_hat = self.p(maps_o)  # shape=(batch_size, bin_bum)
        p_hat = p_hat*p
        KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
        log_p = torch.log(p_hat + 0.00001)
        q = p + 0.00001  # tar
        return KLDivLoss(log_p, q)
    def mse_hist_loss(self):
        return self.emd_loss(self.maps_tar_p, self.maps_out)

    def kl_hist_loss(self):
        return self.kl_loss(self.maps_tar_p, self.maps_out)

