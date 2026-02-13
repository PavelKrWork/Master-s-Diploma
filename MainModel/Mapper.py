import numpy as np

class Mapper:
    """
    Модулятор (mapper) битов в комплексные IQ-символы.
    Поддерживает BPSK, QPSK, 16QAM, 64QAM с Gray-кодированием.
    """
    
    def __init__(self, modulation: str):
        """
        Args:
            modulation: одна из 'BPSK', 'QPSK', '16QAM', '64QAM'
        """
        self.modulation = modulation.upper()
        self.bits_per_symbol = self._bits_per_symbol()
        self.bit_map = self._build_gray_map()

    def _bits_per_symbol(self) -> int:
        return {'BPSK': 1, 'QPSK': 2, '16QAM': 4, '64QAM': 6}[self.modulation]

    def _build_gray_map(self):
        """Строит словарь отображения бит -> символ с Gray-кодированием и нормализацией."""
        bit_map = {}

        if self.modulation == 'BPSK':
            # 0 -> +1, 1 -> -1 (стандартное соглашение)
            bit_map[(0,)] = 1 + 0j
            bit_map[(1,)] = -1 + 0j
            return bit_map

        elif self.modulation == 'QPSK':
            # Gray-код: 00->+1+j, 01->-1+j, 11->-1-j, 10->+1-j
            bits_seq = [(0,0), (0,1), (1,1), (1,0)]
            symbols = [1+1j, -1+1j, -1-1j, 1-1j]
            for bits, sym in zip(bits_seq, symbols):
                bit_map[bits] = sym
            return bit_map

        elif self.modulation == '16QAM':
            # Gray-кодирование для 4 уровней: 00:-3, 01:-1, 11:+1, 10:+3
            gray_levels = {'00': -3, '01': -1, '11': 1, '10': 3}
            symbols = []
            for i_bits in ['00','01','11','10']:
                for q_bits in ['00','01','11','10']:
                    I = gray_levels[i_bits]
                    Q = gray_levels[q_bits]
                    sym = I + 1j*Q
                    bits_tuple = tuple(map(int, i_bits + q_bits))
                    bit_map[bits_tuple] = sym
                    symbols.append(sym)
            # Нормализация к средней мощности 1
            avg_pwr = np.mean(np.abs(symbols)**2)
            for bits in bit_map:
                bit_map[bits] /= np.sqrt(avg_pwr)
            return bit_map

        elif self.modulation == '64QAM':
            # Gray-код для 8 уровней: 000:-7,001:-5,011:-3,010:-1,110:+1,111:+3,101:+5,100:+7
            gray_levels = {
                '000': -7, '001': -5, '011': -3, '010': -1,
                '110':  1, '111':  3, '101':  5, '100':  7
            }
            symbols = []
            for i_bits in ['000','001','011','010','110','111','101','100']:
                for q_bits in ['000','001','011','010','110','111','101','100']:
                    I = gray_levels[i_bits]
                    Q = gray_levels[q_bits]
                    sym = I + 1j*Q
                    bits_tuple = tuple(map(int, i_bits + q_bits))
                    bit_map[bits_tuple] = sym
                    symbols.append(sym)
            avg_pwr = np.mean(np.abs(symbols)**2)
            for bits in bit_map:
                bit_map[bits] /= np.sqrt(avg_pwr)
            return bit_map

        else:
            raise ValueError(f"Неподдерживаемая модуляция: {self.modulation}")

    def map(self, bits) -> np.ndarray:
        """
        Преобразует биты в комплексные IQ-символы.

        Args:
            bits: одномерный массив битов (int 0/1)

        Returns:
            np.ndarray комплексных символов (длина = len(bits) / bits_per_symbol)
        """
        bits = np.asarray(bits).flatten()
        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError(f"Длина битов ({len(bits)}) не кратна {self.bits_per_symbol}")

        n_sym = len(bits) // self.bits_per_symbol
        symbols = np.zeros(n_sym, dtype=complex)

        for i in range(n_sym):
            group = tuple(bits[i*self.bits_per_symbol : (i+1)*self.bits_per_symbol])
            symbols[i] = self.bit_map[group]

        return symbols

    def __repr__(self):
        return f"Mapper(modulation='{self.modulation}')"


mapper = Mapper('BPSK')

bits = np.random.randint(0, 2, 400)

iq_symbols = mapper.map(bits)

print(f"Модуляция: {mapper.modulation}")
print(f"Бит на символ: {mapper.bits_per_symbol}")
print(f"Сгенерировано {len(iq_symbols)} символов")
print(f"Пример первых 5 символов: {iq_symbols[:5]}")