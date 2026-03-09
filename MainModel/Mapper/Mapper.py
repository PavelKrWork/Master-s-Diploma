import numpy as np
class Mapper:
    def __init__(self, constellation):
        self.constellation = constellation
    
    def modulate(self, bits: list):
        bits = np.asarray(bits).flatten()
        n_bits = len(bits)
        bps = self.constellation.bits_per_symbol
        
        if n_bits % bps != 0:
            raise ValueError(f'Number of bits ({n_bits}) must be multiple of {bps}')
        
        n_symbols = n_bits // bps
        bits_reshaped = bits.reshape((n_symbols, bps))
        symbols = []
        
        for group in bits_reshaped:
            key = tuple(group)
            idx = self.constellation.bit_to_idx[key]
            symbols.append(self.constellation.points[idx])
        
        return np.array(symbols)
    
    def get_constellation(self):
        return self.constellation.get_points()