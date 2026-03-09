import sys
sys.path.insert(1, '../Master-s-Diploma/MainModel')


from Mapper.Mapper import Mapper
from Demapper.Demapper import HardDemapper
from Constellation.Constellation import Constellation
import numpy as np

test_bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

print("Initial bits: \n", test_bits)

const_bpsk = Constellation(modulation_type="bpsk")
print("BPSK IQ points: ", const_bpsk.points)
mapper = Mapper(const_bpsk)
symbols_bpsk = mapper.modulate(test_bits)
print("BPSK modulation: \n", symbols_bpsk)
demapper = HardDemapper(const_bpsk)
demod_bpsk_bits = demapper.demodulate(symbols_bpsk)
print("BPSK demodulated bits: \n", demod_bpsk_bits)

print()

const_qpsk = Constellation(modulation_type="qpsk")
print("QPSK IQ points: ", const_qpsk.points)
mapper = Mapper(const_qpsk)
symbols_qpsk = mapper.modulate(test_bits)
print("QPSK modulation: \n", symbols_qpsk)
demapper = HardDemapper(const_qpsk)
demod_qpsk_bits = demapper.demodulate(symbols_qpsk)
print("BPSK demodulated bits: \n", demod_qpsk_bits)

print()

const_16qam = Constellation(modulation_type="16qam")
print("16QAM IQ points: ", const_16qam.points)
mapper = Mapper(const_16qam)
symbols_16qam = mapper.modulate(test_bits)
print("16QAM modulation: \n", symbols_16qam)
demapper = HardDemapper(const_16qam)
demod_16qam_bits = demapper.demodulate(symbols_16qam)
print("BPSK demodulated bits: \n", demod_16qam_bits)

print()

const_64qam = Constellation(modulation_type="64qam")
print("64QAM IQ points: ", const_64qam.points)
mapper = Mapper(const_64qam)
symbols_64qam = mapper.modulate(test_bits)
print("64QAM modulation: \n", symbols_64qam)
demapper = HardDemapper(const_64qam)
demod_64qam_bits = demapper.demodulate(symbols_64qam)
print("BPSK demodulated bits: \n", demod_64qam_bits)