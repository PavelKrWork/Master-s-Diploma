import numpy as np
class DemapperBase:
    def __init__(self, constellation):
        self.constellation = constellation
        self.points = constellation.points
        self.bits_per_symbol = constellation.bits_per_symbol
        self.idx_to_bits = constellation.idx_to_bits
        self.iq_points_num = len(self.points)

class HardDemapper(DemapperBase):
    # поиск наименьшего расстояния от исследуемой точки в received_signal до точек созвездия
    def demodulate(self, received_signal) -> list:
        n_symbols = len(received_signal)
        res_bits = [0] * n_symbols * self.bits_per_symbol
        for ind in range(n_symbols):
            rx_iq_i = received_signal[ind]
            min_dist = float('inf')
            selec_idx = 0
            for idx, const_iq_j in enumerate(self.points):
                dist = np.abs(rx_iq_i - const_iq_j)
                if dist < min_dist:
                    min_dist = dist
                    selec_idx = idx
            
            res_bits[ind * self.bits_per_symbol : (ind + 1) * self.bits_per_symbol] = self.idx_to_bits[selec_idx]

        return res_bits

class SoftDemapper(DemapperBase):
    def demodulate(self, received_signal, noise_variance) -> list:
        pass