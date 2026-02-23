import numpy as np
from Mapper import Modulator

def test_modulator():
    """Тест модулятора: проверка модуляции и нормировки мощности для всех режимов."""
    
    test_cases = [
        ('bpsk', 1, 2),
        ('qpsk', 2, 4),
        ('16qam', 4, 16),
        ('64qam', 6, 64)
    ]
    
    for mode_name, bits_per_sym, const_size in test_cases:
        mod = Modulator(modulation=mode_name)
        
        const = mod.get_constellation()
        assert len(const) == const_size, f"{mode_name}: размер созвездия {len(const)} != {const_size}"
        
        const_power = np.mean(np.abs(const)**2)
        assert np.abs(const_power - 1.0) < 1e-10, f"{mode_name}: мощность созвездия {const_power} != 1"
        
        n_combinations = 2 ** bits_per_sym
        all_bits = []
        for i in range(n_combinations):
            bits = [(i >> (bits_per_sym-1-j)) & 1 for j in range(bits_per_sym)]
            all_bits.extend(bits)
        
        symbols = mod.modulate(all_bits)

        print('Constellation type: ', mode_name)
        print('Modulation symbols: ', symbols)
        print()
        
        assert len(symbols) == n_combinations, f"{mode_name}: число символов {len(symbols)} != {n_combinations}"
        
        symbol_power = np.mean(np.abs(symbols)**2)
        assert np.abs(symbol_power - 1.0) < 0.01, f"{mode_name}: мощность сигнала {symbol_power} != 1"
        
        assert len(np.unique(symbols)) == n_combinations, f"{mode_name}: символы не уникальны"
        
        symbol_to_bits = {}
        for i, sym in enumerate(const):
            for bits_tuple, idx in mod._bit_to_idx.items():
                if idx == i:
                    symbol_to_bits[sym] = np.array(bits_tuple)
                    break
        
        for i, sym in enumerate(symbols):
            closest_idx = np.argmin(np.abs(sym - const))
            closest_bits = symbol_to_bits[const[closest_idx]]
            original_bits = np.array(all_bits[i*bits_per_sym:(i+1)*bits_per_sym])
            assert np.array_equal(original_bits, closest_bits), f"{mode_name}: биты не совпадают"


if __name__ == "__main__":
    test_modulator()