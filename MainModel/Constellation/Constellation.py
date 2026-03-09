import numpy as np

class Constellation:
    def __init__(self, modulation_type='qpsk'):
        self.modulation_type = modulation_type.lower()
        self._init_constellation()
    
    def _init_constellation(self):
        if self.modulation_type == 'bpsk':
            self.points = np.array([-1.0, 1.0], dtype=complex)
            self.bits_per_symbol = 1
            self.bit_to_idx = { (0,):0, (1,):1 }
            self.idx_to_bits = {0: [0], 1: [1]}
            
        elif self.modulation_type == 'qpsk':
            s = np.array([1+1j, 1-1j, -1-1j, -1+1j]) / np.sqrt(2)
            self.points = s
            self.bits_per_symbol = 2
            self.bit_to_idx = {
                (0,0):0, (0,1):1, (1,1):2, (1,0):3
            }
            self.idx_to_bits = {
                0: [0,0], 1: [0,1], 2: [1,1], 3: [1,0]
            }
            
        elif self.modulation_type == '16qam':
            pam4 = np.array([-3, -1, 1, 3])
            gray = [0,1,3,2]
            scale = np.sqrt(10.0)
            points = []
            bit_to_idx = {}
            idx_to_bits = {}
            idx = 0
            for i in range(4):   # Q
                for r in range(4): # I
                    points.append(complex(pam4[r], pam4[i]) / scale)
                    i_bits = [(gray[i]>>1)&1, gray[i]&1]
                    r_bits = [(gray[r]>>1)&1, gray[r]&1]
                    bits_tuple = tuple(r_bits + i_bits)
                    bit_to_idx[bits_tuple] = idx
                    idx_to_bits[idx] = list(bits_tuple)
                    idx += 1
            self.points = np.array(points)
            self.bits_per_symbol = 4
            self.bit_to_idx = bit_to_idx
            self.idx_to_bits = idx_to_bits
            
        elif self.modulation_type == '64qam':
            pam8 = np.array([-7,-5,-3,-1,1,3,5,7])
            gray = [0,1,3,2,6,7,5,4]
            scale = np.sqrt(42.0)
            points = []
            bit_to_idx = {}
            idx_to_bits = {}
            idx = 0
            for i in range(8):   # Q
                for r in range(8): # I
                    points.append(complex(pam8[r], pam8[i]) / scale)
                    i_bits = [(gray[i]>>2)&1, (gray[i]>>1)&1, gray[i]&1]
                    r_bits = [(gray[r]>>2)&1, (gray[r]>>1)&1, gray[r]&1]
                    bits_tuple = tuple(r_bits + i_bits)
                    bit_to_idx[bits_tuple] = idx
                    idx_to_bits[idx] = list(bits_tuple)
                    idx += 1
            self.points = np.array(points)
            self.bits_per_symbol = 6
            self.bit_to_idx = bit_to_idx
            self.idx_to_bits = idx_to_bits
            
        else:
            raise ValueError(f'Unsupported modulation: {self.modulation_type}')
    
    def get_points(self):
        return self.points.copy()