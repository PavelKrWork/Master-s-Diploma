import numpy as np

class Modulator:
    def __init__(self, modulation='qpsk'):
        self.modulation = modulation.lower()
        self._init_constellation()

    def _init_constellation(self):
        if self.modulation == 'bpsk':
            self.constellation = np.array([-1.0, 1.0], dtype=complex)
            self.bits_per_symbol = 1
            self._bit_to_idx = { (0,):0, (1,):1 }

        elif self.modulation == 'qpsk':
            # Gray mapping: 00->1+j, 01->1-j, 11->-1-j, 10->-1+j
            s = np.array([1+1j, 1-1j, -1-1j, -1+1j]) / np.sqrt(2)
            self.constellation = s
            self.bits_per_symbol = 2
            self._bit_to_idx = {
                (0,0):0, (0,1):1, (1,1):2, (1,0):3
            }

        elif self.modulation == '16qam':
            # 4x4 Gray mapping, нормировка на sqrt(10)
            pam4 = np.array([-3, -1, 1, 3])
            gray = [0,1,3,2]  # индексы для PAM-4 с Греем
            scale = np.sqrt(10.0)
            const = []
            mapping = {}
            idx = 0
            for i in range(4):   # Q
                for r in range(4): # I
                    const.append(complex(pam4[r], pam4[i]) / scale)
                    i_bits = [(gray[i]>>1)&1, gray[i]&1]
                    r_bits = [(gray[r]>>1)&1, gray[r]&1]
                    bits = tuple(r_bits + i_bits)
                    mapping[bits] = idx
                    idx += 1
            self.constellation = np.array(const)
            self.bits_per_symbol = 4
            self._bit_to_idx = mapping

        elif self.modulation == '64qam':
            # 8x8 Gray mapping, нормировка на sqrt(42)
            pam8 = np.array([-7,-5,-3,-1,1,3,5,7])
            gray = [0,1,3,2,6,7,5,4]  # 3-битный Грей
            scale = np.sqrt(42.0)
            const = []
            mapping = {}
            idx = 0
            for i in range(8):   # Q
                for r in range(8): # I
                    const.append(complex(pam8[r], pam8[i]) / scale)
                    i_bits = [(gray[i]>>2)&1, (gray[i]>>1)&1, gray[i]&1]
                    r_bits = [(gray[r]>>2)&1, (gray[r]>>1)&1, gray[r]&1]
                    bits = tuple(r_bits + i_bits)
                    mapping[bits] = idx
                    idx += 1
            self.constellation = np.array(const)
            self.bits_per_symbol = 6
            self._bit_to_idx = mapping

        else:
            raise ValueError(f'Unsupported modulation: {self.modulation}')

    def modulate(self, bits):
        bits = np.asarray(bits).flatten()
        n_bits = len(bits)
        n_symbols = n_bits // self.bits_per_symbol
        if n_bits % self.bits_per_symbol != 0:
            raise ValueError('Number of bits must be multiple of bits_per_symbol')
        bits_reshaped = bits.reshape((n_symbols, self.bits_per_symbol))
        symbols = []
        for group in bits_reshaped:
            key = tuple(group)
            symbols.append(self.constellation[self._bit_to_idx[key]])
        return np.array(symbols)

    def get_constellation(self):
        return self.constellation.copy()