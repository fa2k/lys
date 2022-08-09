import time
import sys
import struct
import socket
from dataclasses import dataclass
from tracemalloc import start
import colorsys
import numpy as np
from numpy.random import default_rng
rng = default_rng()


lasttime = time.time()
target_fps = 100



def make_packet(data, universe):
    """Get ArtNet packet for channel data."""

    return struct.pack(
            "<8sHBB"    # Header, opcode, protocol version
            "BBHBB"     # Sequence, physical, universe, len (hi,lo)
            f"{len(data)}s", # data
            "Art-Net".encode('ascii'), 0x5000, 0, 14,
            0, 0, universe,
            len(data) >> 8, len(data) & 0xff,
            data)


output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

@dataclass
class ArtNetNode:
    address: str
    universe: int
    buffer_index: int
    data_len: int
    port: int = 6454
    # Skip this many frames between sending to reduce frame rate
    frame_skipping: int = 0


output_nodes = [
    # Taklys totalt 895 RGB pixels
    ArtNetNode("192.168.1.224", 1,    0, 510),
    ArtNetNode("192.168.1.224", 2,  510, 510),
    ArtNetNode("192.168.1.224", 3, 1020, 510),
    ArtNetNode("192.168.1.224", 4, 1530, 510),
    ArtNetNode("192.168.1.224", 5, 2040, 510),
    ArtNetNode("192.168.1.224", 6, 2550, 135),
    # Hyllelys 3 pixels
    ArtNetNode("192.168.1.226", 0, 2685,   9, frame_skipping=2),
]
outputs_frame_skipping = np.zeros(len(output_nodes))

buffer_size = sum(node.data_len for node in output_nodes)
num_pixels = buffer_size // 3

# Define 3D position of each pixel
physical_positions = np.zeros((num_pixels, 3), dtype=np.float64)

def make_segment(positions_array, start_index, start_point, end_point, count):
    #for i in range(count):
    startnp = np.array(start_point, dtype='float64', ndmin=2)
    endnp = np.array(end_point, dtype='float64', ndmin=2)
    positions_array[start_index:start_index+count, :] =  startnp + \
                                                (endnp - startnp) * np.linspace(0, 1, num=count).reshape((count, 1))

make_segment(physical_positions, 0,   [0,0.1,3], [0,4,3], 224)
make_segment(physical_positions, 224, [1,0.1,3], [1,4,3], 224)
make_segment(physical_positions, 448, [2,0.1,3], [2,4,3], 224)
make_segment(physical_positions, 672, [3,0.1,3], [3,4,3], 223)

make_segment(physical_positions, 895, [2.5,0,1], [3,0,1], 3)

# Visual effects generation

class Para:
    """Time-dependent parameter interface."""
    def get(self, t : float):
        return 0


class Const(Para):
    """Parameter that does not depend on time."""

    def __init__(self, data : np.array):
        self.data = data

    def get(self, t):
        return self.data


def make_color(r : float, g : float, b : float):
    return Const(np.array([r,g,b]))


class Layer:
    """Layer interface is like Para, but supports replacing itself.
    The return value of get_layer should be a tuple (data, next_layer)
    where next_layer is the layer to use next time."""
    
    def get_layer(self, t : float):
        pass



class Clouds(Layer):
    pass


class Fill(Layer):
    """Generate a specific colour."""

    def __init__(self, size_in_pixels : int, color : Para):
        self.size = size_in_pixels
        self.color = color

    def get_layer(self, t):
        return (np.ones((self.size, 3)) * self.color.get(t), self)


class CircleFader:

    def get(self, t):
        return 0


class GradientFader:
    pass


class GlobalFader(Layer):
    """Fade all pixels the same way between two Layer objects."""

    def __init__(self, 
                start_time : float, interval : float,
                from_layer : Layer, to_layer : Layer,
                mode = 'linear'):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.start_time = start_time
        self.interval = interval
        self.mode = mode
    
    def get_layer(self, t):
        (from_data, self.from_layer) = self.from_layer.get_layer(t)
        (to_data, self.to_layer) = self.to_layer.get_layer(t)
        progress = max((t - self.start_time) / self.interval, 0)
        if progress >= 1:
            return (to_data, self.to_layer)
        elif self.fade_mode == 'linear':
            return (to_data * (1-progress) + from_data * progress, self)
        else:
            raise ValueError(f"Invalid fade mode {self.fade_mode}.")

    def next(self):
        if self.done:
            return self.to_layer
        else:
            return self


class RgbGamma:
    """Empirical gamma curve to make RGB lights nicer."""

    def __init__(self, parent : Layer):
        self.parent = parent
    
    def get_layer(self, t):
        (data, self.parent) = self.parent.get_layer(t)
        return data ** 2.1


class StableSpatialDithering:
    """Class to turn float RGB values in [0,1] into integers in [0,255].
    It implements a random rounding effect, but tries to achieve the
    following:
    * The overall average values across all channels of a colour in the
    output are close to the target overall values.
    * The rounding of any particular pixels stays the same in subsequent
    calls.

    It is not a Layer, but intended to run as the last step before generating
    packets. It is not aware of physical positions, only operates on global
    averages.
    """
    
    def __init__(self, layer : Layer, rng=None):
        self.layer = layer
        self.last_outputs = np.array(0)
        self.last_stresses = np.array(0)
        self.rng = rng or np.random.default_rng()

    def next(self, t):
        (data, self.layer) = self.layer.get_layer(t)
        
        targets = data * 255

        # If the outputs are now wrong by >1 unit, plus or minus, we
        # just start over and discard any stress/state. The 
        # "last outputs" are modified so the following algorithm can
        # randomise the rounding, from scratch.
        is_wrong_output = np.abs(targets - self.last_outputs) > 1
        if len(self.last_outputs.shape) > 0:    
            self.last_outputs[is_wrong_output] = np.floor(targets)
            self.last_stresses[is_wrong_output] = 0

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
            interval_remain * self.rng.random() < stress_increases
        )

        new_values = self.last_outputs + increments
        self.last_outputs = new_values
        self.last_stresses = stresses
        return new_values.flatten()



#-----------SETUP-----------

scene = Fill(num_pixels, make_color(20, 20, 20))


channel_data_generator = StableSpatialDithering(scene)


#---------------------------




preview_mode = True

if preview_mode:
    import matplotlib.pyplot as plt
    plt.ion()
    #ig = plt.figure()
    
    xmax, ymax = physical_positions[:, 0].max(), physical_positions[:, 1].max()
    xpix, ypix = 250, 250
    
    image = np.zeros((xpix+1, ypix+1, 3), dtype=np.uint8)

    xindexes = np.array([[(0.1 + (pp[0]/xmax)*0.8) * xpix]*3 for pp in physical_positions], dtype='uint64').reshape(-1)
    yindexes = np.array([[(0.1 + (pp[1]/ymax)*0.8) * ypix]*3 for pp in physical_positions], dtype='uint64').reshape(-1)
    cindexes = np.array([0,1,2] * len(physical_positions), dtype='uint64').reshape(-1)

    def show_preview(output_buffer):
        image[xindexes, yindexes, cindexes] = output_buffer
        plt.imshow(image)
        plt.draw()
        plt.pause(0.0001)
        #plt.clf()
        #plt.show()

for frame in range(10):
    t = time.time()

    output_buffer = channel_data_generator.next(t)

    for i, node in enumerate(output_nodes):
        if node.frame_skipping:
            if outputs_frame_skipping[i] == node.frame_skipping:
                outputs_frame_skipping[i] = 0
            else:
                outputs_frame_skipping[i] += 1
                continue

        packet_data = make_packet(
            output_buffer[node.buffer_index:node.buffer_index+node.data_len].tobytes(),
            node.universe
            )
        #output_socket.sendto(packet_data, (node.address, node.port))

    if preview_mode:
        show_preview(output_buffer)

    dt = t - lasttime
    time.sleep(max(0, 1 / target_fps - dt))
    lasttime = t
