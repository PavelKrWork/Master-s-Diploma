import sys
sys.path.insert(1, '../Master-s-Diploma/MainModel')


from Mapper.Mapper import Mapper
from Constellation.Constellation import Constellation
import numpy as np

test_bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

const_bpsk = Constellation(modulation_type="bpsk")
mapper = Mapper(const_bpsk)
symbols_bpsk = mapper.modulate(test_bits)
print("BPSK modulation: \n", symbols_bpsk)

print()

const_qpsk = Constellation(modulation_type="qpsk")
mapper = Mapper(const_qpsk)
symbols_qpsk = mapper.modulate(test_bits)
print("QPSK modulation: \n", symbols_qpsk)

print()

const_16qam = Constellation(modulation_type="16qam")
mapper = Mapper(const_16qam)
symbols_16qam = mapper.modulate(test_bits)
print("16QAM modulation: \n", symbols_16qam)

print()

const_64qam = Constellation(modulation_type="64qam")
mapper = Mapper(const_64qam)
symbols_64qam = mapper.modulate(test_bits)
print("64QAM modulation: \n", symbols_64qam)