import numpy as np

class OutputAdapter:
    """Class to turn float RGB values (2D array) in [0,1] into a flat arrray of
    integers in [0,255].

    OutputAdapters are not Layers, but intended to run as the last step before generating
    packets. They are not aware of physical positions.
    """
    def next(self, data):
        pass


class StableSpatialDithering(OutputAdapter):
    """
    Implements a random rounding effect, but tries to achieve the following:
    * The overall average values across all channels of a colour in the
    output are close to the target overall values.
    * The rounding of any particular pixels stays the same in subsequent
    calls.
    """
    
    def __init__(self, rng=None):
        self.last_outputs = np.array(0)
        self.last_stresses = np.array(0)
        self.rng = rng or np.random.default_rng()

    def next(self, data):
        targets = data * 255

        # If the outputs are now wrong by >1 unit, plus or minus, we
        # just start over and discard any stress/state. The 
        # "last outputs" are modified so the following algorithm can
        # randomise the rounding, from scratch.
        is_wrong_output = np.abs(targets - self.last_outputs) > 1
        if len(self.last_outputs.shape) > 0:
            np.putmask(self.last_outputs, is_wrong_output, np.floor(targets))
            np.putmask(self.last_stresses, is_wrong_output, 0)
        else:#dbg
            self.last_outputs = np.zeros(targets.shape)

        stresses = targets - self.last_outputs
        # Determine if the outputs should change when the target changes. 
        # Probability of changing the output, if the target changes in the
        # same direction as the last stress:
        # P(change) = (stresse - last_stresses) / interval_remain
        stress_increases = np.maximum(np.sign(stresses) * (stresses - self.last_stresses), 0)

        # Remaining interval until crossing over to a new integer value,
        # taken relative to the last outputs.
        interval_remain = 1 - np.abs(self.last_stresses)

        increments = np.sign(stresses) * (
            interval_remain * self.rng.random(targets.shape) < stress_increases
        )

        new_values = self.last_outputs + increments
        self.last_outputs = new_values
        self.last_stresses = targets - new_values
        return new_values.flatten().astype(np.uint8)


class GlobalDimmerFourByte(OutputAdapter):
    """
    Output four bytes per pixel, with a five-bit global dimmer as the first byte.
    This provides excellent dynamic range, so dithering is not necessary.
    """
    
    def __init__(self):
        pass

    def next(self, data):
        maxes = np.max(data, axis=1)
        dimmers = np.maximum(1, np.ceil(31 * maxes))
        rgb = np.round(data * 255 * 31 / dimmers.reshape((-1, 1)))
        return np.hstack([dimmers.reshape((-1, 1)), rgb]).astype(np.uint8).flatten()

