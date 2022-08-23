from code import InteractiveInterpreter
from operator import is_
import time
import sys
import struct
import socket
from dataclasses import dataclass
from tracemalloc import start
import colorsys
from turtle import speed
import numpy as np
from numpy.random import default_rng
from paho.mqtt import client as mqtt
import threading


rng = default_rng()


lasttime = time.time()
TARGET_FPS = 100
# Reduce fps when output doesn't change?
STATIC_OUTPUT_SKIP_N_FRAMES = 10000


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
make_segment(physical_positions, 224, [1,4,3], [1,0.1,3], 224)
make_segment(physical_positions, 448, [2,0.1,3], [2,4,3], 224)
make_segment(physical_positions, 672, [3,4,3], [3,0.1,3], 223)

make_segment(physical_positions, 895, [2,0,1], [3,0,1], 3)

#make_segment(physical_positions, 898, [2,0,0.5], [3,0,0.5], 1)

# Visual effects generation

def make_color_from_hsv(h : float, s : float, v : float):
    return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)


class Layer:
    """Layer interface is like Para, but supports replacing itself.
    The return value of get_layer should be a tuple (data, next_layer)
    where next_layer is the layer to use next time."""
    
    def get_layer(self, t : float):
        pass


class ScrollingRainbow(Layer):
    """Scrolling color gradient along an axis.
    It's not a gradient, but varies the hue of the colours along an axis.
    The length of the scrolling vector determines the physical size of
    one period - 360 degrees of hue. The scrolling_period determines the
    number of seconds for one temporal period.
    """
    def __init__(self, physical_positions: np.array,
                    scrolling_vector : np.array,
                    scrolling_period : float,
                    brightness : np.array
                    ):
        self.brightness = brightness
        self.scrolling_period = scrolling_period
        self.physical_phase = np.dot(physical_positions, scrolling_vector)
    
    def get_layer(self, t):
        tt = t / self.scrolling_period - t // self.scrolling_period
        hues = ((self.physical_phase + tt) * 360) % 360
        rgb_data = np.stack([
            120 - np.minimum(hues, 360-hues),
            120 - np.abs(120 - hues),
            120 - np.abs(240 - hues)
        ], axis=1).clip(0, 60) * (self.brightness / 60)
        return (rgb_data, self)


class PlanarSplit(Layer):
    def __init__(self, physical_positions : np.array,
                intercept : np.array, normal : np.array,
                under : Layer, over : Layer):
        
        shifted = physical_positions - intercept
        # take dot product of positions and normal vector norm
        self.mask = (np.dot(shifted, normal) > 0).reshape((len(physical_positions), 1))
        self.under = under
        self.over = over
    
    def get_layer(self, t : float):
        (dn, self.under) = self.under.get_layer(t)
        (up, self.over) = self.over.get_layer(t)
        return (np.where(self.mask, up, dn), self)


class Clouds(Layer):
    pass


class Fill(Layer):
    """Generate a specific colour."""

    def __init__(self, size_in_pixels : int, color):
        self.buffer = np.ones((size_in_pixels, 3)) * color

    def get_layer(self, t):
        return (self.buffer, self)


class RadialFader(Layer):
    """Fade all pixels the in a spherical pattern."""

    def __init__(self, 
                physical_positions,
                start_time : float,
                center_point : np.array, speed_meters_per_second : float,
                from_layer : Layer, to_layer : Layer,
                soft : float = 0.2,
                direction_out : bool = True 
                ):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.transition_time = soft
        distances = np.sqrt(np.sum((physical_positions - center_point) ** 2, axis=1))
        self.time_offsets = (distances / speed_meters_per_second) + start_time
        max_time = np.max(self.time_offsets)
        if not direction_out:
            self.time_offsets = max_time - self.time_offsets + start_time
        self.end_time = max_time + self.transition_time
    
    def get_layer(self, t):
        (to_data, self.to_layer) = self.to_layer.get_layer(t)
        if t > self.end_time:
            return (to_data, self.to_layer)
        (from_data, self.from_layer) = self.from_layer.get_layer(t)
        progresses = np.clip((t - self.time_offsets) / self.transition_time, 0, 1).reshape((to_data.shape[0], 1))
        return (to_data * progresses + from_data * (1-progresses), self)


class LinearGradient(Layer):
    """Constant gradient spatial fader with no time dependence.
    The length of fade_vector determines the sharpness of the transition.
    If fade_vector is 2 m, then the fade spans 0.5 m."""

    def __init__(self, 
                physical_positions,
                fade_start_point : np.array,
                fade_vector : np.array,
                from_layer : Layer, to_layer : Layer
                ):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.fade_amounts = np.clip(
            np.dot((physical_positions - fade_start_point), fade_vector),
            0, 1).reshape((len(physical_positions), 1))
    
    def get_layer(self, t):
        (to_data, self.to_layer) = self.to_layer.get_layer(t)
        (from_data, self.from_layer) = self.from_layer.get_layer(t)
        return (to_data * self.fade_amounts + from_data * (1-self.fade_amounts), self)


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
        elif self.mode == 'linear':
            return (to_data * progress + from_data * (1-progress), self)
        else:
            raise ValueError(f"Invalid fade mode {self.fade_mode}.")

    def next(self):
        if self.done:
            return self.to_layer
        else:
            return self


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



#-----------SETUP-----------

# Root scene controller manages the layer tree
# Custom undocumented code is just for me
class SceneController:
    def __init__(self, num_pixels, start_color_hsv=(0.8, 0.5, 0.3), scene_gamma=1):
        self.is_on = False
        self.mode = None
        self.hsv = start_color_hsv
        self.on_layer = Fill(num_pixels, make_color_from_hsv(*start_color_hsv))
        self.off_layer = Fill(num_pixels, np.array([0, 0, 0]))
        self.root = self.off_layer
        self.num_pixels = num_pixels
        self.scene_gamma = scene_gamma
        self.lock = threading.Lock()
        self._apply_new_state(True, start_color_hsv, "normal")
    
    def set_color(self, str_payload):
        if str_payload == "OFF":
            new_power_state = False
            new_color = self.hsv
        else:
            new_power_state = True
            new_color = list(self.hsv)
            if str_payload == "ON":
                pass
            if str_payload == "INCREASE":
                new_color[2] = min(1.0, new_color[2] + 0.01)
            elif str_payload == "DECREASE":
                new_color[2] = max(0.0, new_color[2] - 0.01)
            elif str_payload == "ON":
                pass
            else:
                parts = str_payload.split(",")
                try:
                    if len(parts) == 3:
                        h2 = float(parts[0]) / 360.0
                        s2 = float(parts[1]) / 100.0
                    else:
                        h2, s2 = new_color[:2]
                    if len(parts) == 1 or len(parts) == 3:
                        v2 = float(parts[-1]) / 100.0
                        new_color = [max(0, min(1, x)) for x in (h2, s2, v2)]
                except ValueError as e:
                    print(f"Recevied invalid color {str_payload}: '{e}'.")
        if new_color[2] == 0.0:
            new_power_state = False
            new_color[2] = self.hsv[2] # Don't set color if blacking
        self._apply_new_state(new_power_state, new_color, self.mode)

    def set_mode(self, mode):
        self._apply_new_state(self.is_on, self.hsv, mode)

    def set_power(self, power):
        self._apply_new_state(power == "ON", self.hsv, self.mode)

    def _apply_new_state(self, new_power_state, new_color, new_mode):
        #print("Applying", new_power_state, ",", new_color, ",", new_mode)
        if new_power_state != self.is_on or new_color != self.hsv or new_mode != self.mode:
            if not new_power_state:
                new_layer = self.off_layer
            else:
                if new_mode == "normal": #TODO clouds
                    new_layer1 = Fill(self.num_pixels, make_color_from_hsv(*new_color))
                    new_layer2 = ScrollingRainbow(physical_positions, np.array([0.5, 0, 0]), 20, new_color[-1])
                    new_layer = PlanarSplit(physical_positions, np.array([0,0,2]), np.array([0,0,1]),
                                new_layer2, new_layer1)
                if new_mode == "night": #TODO stars
                    new_layer1 = Fill(self.num_pixels, make_color_from_hsv(*new_color))
                    new_layer2 = ScrollingRainbow(physical_positions, np.array([0.5, 0, 0]), 20, new_color[-1])
                    new_layer = PlanarSplit(physical_positions, np.array([0,0,2]), np.array([0,0,1]),
                                new_layer2, new_layer1)
                elif new_mode == "basic":
                    new_layer1 = Fill(self.num_pixels, make_color_from_hsv(*new_color))
                    new_layer2 = ScrollingRainbow(physical_positions, np.array([0.5, 0, 0]), 20, new_color[-1])
                    new_layer = PlanarSplit(physical_positions, np.array([0,0,2]), np.array([0,0,1]),
                                new_layer2, new_layer1)
                elif new_mode == "cinema":
                    new_layer = LinearGradient(physical_positions, 
                                            np.array([0, 4, 0]),
                                            np.array([0, -1, 0]),
                                            Fill(self.num_pixels, make_color_from_hsv(*new_color)),
                                            self.off_layer)
        else: # No change
            return
        with self.lock:
            if "cinema" in [new_mode, self.mode]:
                self.root = GlobalFader(time.time(), 4, self.root, new_layer)
            elif self.is_on != new_power_state:
                if new_mode == "night": #start fade from bedroom
                    center = np.array([0, 2.0, 3.0])
                    speed = 2
                else:
                    center = np.array([1.5, 2.0, 3.0])
                    speed = 4
                self.root = RadialFader(physical_positions, time.time(), 
                                    center, speed,
                                    self.root, new_layer, soft=1,
                                    direction_out=new_power_state)
            elif self.mode == new_mode: # just a colour change
                self.root = GlobalFader(time.time(), 1, self.root, new_layer)
            else: # fallback
                self.root = GlobalFader(time.time(), 1, self.root, new_layer)

        if new_power_state:
            self.on_layer = new_layer
        self.is_on = new_power_state
        self.hsv = new_color
        self.mode = new_mode

    def get_data(self, t):
        with self.lock:
            (data, self.root) = self.root.get_layer(t)
        #if self.mask_state:
        #    data[self.mask_pixels,:] = np.array(self.hsv)
        #else:
        #    data[self.mask_pixels,:] = 0
        return data ** self.scene_gamma


scene = SceneController(num_pixels, start_color_hsv=(0.5, 0.5, 0.5), scene_gamma=2.1)
output_dithering_adapter = StableSpatialDithering()

#----- MQTT Command loop --------
MQTT_HOST = "192.168.1.8"
MQTT_TOPIC_BASE = "stue/lys"
FADE_TIME = 500 / 1000.0


client = mqtt.Client()

def on_connect(client, _, flags, rc):
    client.subscribe(f"{MQTT_TOPIC_BASE}/#")
client.on_connect = on_connect

def on_message(client, state, msg):
    try:
        str_payload = msg.payload.decode('ascii')
    except ValueError:
        return
    if str_payload == "EXIT":
        sys.exit(0)
    topic = msg.topic[len(f"{MQTT_TOPIC_BASE}/"):]
    #print(topic, "|", str_payload)
    if topic == "color":
        scene.set_color(str_payload)
    elif topic == "mode":
        scene.set_mode(str_payload)
    elif topic == "power":
        scene.set_power(str_payload)
    #elif topic == "mask1":
    #    scene.set_masking(str_payload)

client.on_message = on_message
client.connect(MQTT_HOST, 1883, 60)
client.loop_start()


#---------------------------

preview_mode = False

if preview_mode:
    import pygame
    import pygame.surfarray as surfarray
    pygame.init()
    xmax, ymax = physical_positions[:, 0].max(), physical_positions[:, 1].max()
    xpix, ypix = 500, 500
    screen = pygame.display.set_mode((xpix, ypix))
    
    image = np.zeros((xpix, ypix, 3), dtype=np.uint8)

    xindexes = np.array(xmax - [[(0.1 + (pp[0]/xmax)*0.8) * xpix]*3 for pp in physical_positions], dtype='uint64').reshape(-1)
    yindexes = np.array([[(0.1 + (pp[1]/ymax)*0.8) * ypix]*3 for pp in physical_positions], dtype='uint64').reshape(-1)
    cindexes = np.array([0,1,2] * len(physical_positions), dtype='uint64').reshape(-1)

    def show_preview(output_buffer):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        image[xindexes, yindexes, cindexes] = output_buffer
        surfarray.blit_array(screen, image)
        pygame.display.flip()

prev_output_buffer = np.zeros([1])
equal_frames_count = 0


# ----- FRAME LOOP -----

#for frame in range(100):
while True:
    t = time.time()

    output_buffer = output_dithering_adapter.next(scene.get_data(t))
    #output_buffer = (scene.get_data(t) * 255).astype(np.uint8).flatten()
    do_send_output = True
    if np.array_equal(output_buffer, prev_output_buffer):
        equal_frames_count += 1
        if equal_frames_count < STATIC_OUTPUT_SKIP_N_FRAMES:
            do_send_output = False
        else:
            equal_frames_count = 0
    else:
        equal_frames_count = 0
    prev_output_buffer = output_buffer

    if do_send_output:
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
            if not preview_mode:
                output_socket.sendto(packet_data, (node.address, node.port))

        if preview_mode:
            show_preview(output_buffer)

    dt = t - lasttime
    time.sleep(max(0, 1 / TARGET_FPS - dt))
    lasttime = t

