class DemapperИase:
    def __init__(self, constellation_type, bits_per_symbol):
        pass
   
    def _generate_constellation(self):
        pass
    
    def _generate_mapping(self):
        pass


class HardDemapper(Demapper):
    def demodulate(self, received_signal):
        pass

class SoftDemapper(Demapper):
    def demodulate(self, received_signal, noise_variance):
        pass