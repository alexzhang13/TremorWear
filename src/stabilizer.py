

class Stabilizer():
    def __init__(self):
        pass

    def calibrate(self):
        pass

# Stabilizer for Moment of Inertia
class StabilizerMOI(Stabilizer):
    def __init__(self):
        super(StabilizerMOI, self).__init__()
        self.moment_of_i = 0

    def calibrate(self):
        pass

# Stabilizer Function for Processing and Actuator System Delays
class StabilizerD(Stabilizer):
    def __init__(self):
        super(StabilizerD, self).__init__()
        self.delay = 0
        self.loss = 0

    def calibrate(self):
        pass