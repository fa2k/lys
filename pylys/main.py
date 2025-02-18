from code import InteractiveInterpreter
from operator import is_
import time
import sys
import struct
import socket
import math
from dataclasses import dataclass
from collections.abc import Sequence
import colorsys
from sklearn import linear_model
import numpy as np
from numpy.random import default_rng
from paho.mqtt import client as mqtt
import threading
from output_adapters import *

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
    pixel_index: int
    num_pixels: int
    output_adapter: object
    port: int = 6454
    # Skip this many frames between sending to reduce frame rate
    frame_skipping: int = 0

# Define Art-Net universes. The parameters specify the pixels to output. The pixel data is converted
# to bytes by the output_adapter.
output_nodes = [
    # Taklys totalt 901 RGB pixels @ 4 bytes per pixel
    ArtNetNode("192.168.1.224", 1,    0, 128, GlobalDimmerFourByte()),
    ArtNetNode("192.168.1.224", 2,  128, 128, GlobalDimmerFourByte()),
    ArtNetNode("192.168.1.224", 3,  256, 128, GlobalDimmerFourByte()),
    ArtNetNode("192.168.1.224", 4,  384, 128, GlobalDimmerFourByte()),
    ArtNetNode("192.168.1.224", 5,  512, 128, GlobalDimmerFourByte()),
    ArtNetNode("192.168.1.224", 6,  640, 128, GlobalDimmerFourByte()),
    ArtNetNode("192.168.1.224", 7,  768, 128, GlobalDimmerFourByte()),
    ArtNetNode("192.168.1.224", 8,  896,   5, GlobalDimmerFourByte()),
    # Hyllelys 3 pixels
    ArtNetNode("192.168.1.226", 0,  901,   3, StableSpatialDithering(), frame_skipping=2),
]
outputs_frame_skipping = np.zeros(len(output_nodes))

num_pixels = sum([node.num_pixels for node in output_nodes])

# Define 3D position of each pixel
physical_positions = np.zeros((num_pixels, 3), dtype=np.float64)

def make_segment(positions_array, start_index, start_point, end_point, count):
    #for i in range(count):
    startnp = np.array(start_point, dtype='float64', ndmin=2)
    endnp = np.array(end_point, dtype='float64', ndmin=2)
    positions_array[start_index:start_index+count, :] =  startnp + \
                                                (endnp - startnp) * np.linspace(0, 1, num=count).reshape((count, 1))

make_segment(physical_positions, 0,   [0,0.1,3], [0,4,3], 226)
make_segment(physical_positions, 226, [1,4,3], [1,0.1,3], 225)
make_segment(physical_positions, 451, [2,0.1,3], [2,4,3], 224)
make_segment(physical_positions, 675, [3,4,3], [3,0.1,3], 226)

make_segment(physical_positions, 901, [2,0,1], [3,0,1], 3)

#make_segment(physical_positions, 898, [2,0,0.5], [3,0,0.5], 1)

# Visual effects generation

def hsv_to_rgb_array(h : float, s : float, v : float):
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


# TODO opensimplex.seed(1337)
class Clouds(Layer):
    """TODO clouds doesn't work - stuck in opensimplex"""
    starttime = 0

    def __init__(self, physical_positions : np.array, rgb : np.array):
        self.physical_positions = physical_positions / physical_positions.max(axis=0)
        self.rgb = rgb
        self.buffer = np.zeros(self.physical_positions.shape)

    def get_layer(self, t: float):
        if Clouds.starttime == 0:
            Clouds.starttime = t
        tpar = (t - Clouds.starttime) / (24*60*60)

        self.buffer[:, 2] = self.rgb[2]
        print(self.physical_positions[:, 0].shape)
        self.whiteclouds = opensimplex.noise3array(
            self.physical_positions[:, 0],
            self.physical_positions[:, 1],
            self.physical_positions[:, 2] + tpar
        )
        minrg = min(*self.rgb[0:2])
        self.buffer[:, 0:2] = (self.whiteclouds + 1) * minrg
        return (self.buffer, self)

class WildWaves(Layer):
    """Happy accident during development, all kinds of strange things going on."""
    def __init__(self, physical_positions : np.array):
        target_average_color=np.array((0.5, 0.5, 1.0))
        background_color=np.array([0.1, 0.1, 1.0])
        wave_colors=np.array([
            [1.0, 0.2, 0.0],
            [0.5, 0.5, 0.5]
        ]).transpose()
        wave_vectors=np.array([
            [2, 0, 0],
            [0.5, 1.5, 0]
        ]).transpose()
        wave_speeds=np.array([0.5, 0.2])

        self.colors = np.hstack([wave_colors, background_color.reshape(3, 1)])
        self.wave_vectors = wave_vectors
        coefficients = linear_model.Ridge(alpha=.5, fit_intercept=False) \
                                    .fit(self.colors, target_average_color) \
                                    .coef_
        self.wave_means = coefficients
        self.wave_amplitudes = (1-coefficients)
        dot_products = np.matmul(physical_positions, wave_vectors)
        phys_lengths = np.linalg.norm(physical_positions, axis=1, keepdims=True)
        wave_vector_sq_lengths = np.sum(np.square(wave_vectors), axis=0, keepdims=True)
        self.positional_phases = dot_products / (phys_lengths * wave_vector_sq_lengths)
        self.wave_speeds = wave_speeds

    def get_layer(self, t: float):
        # Set to background
        waves = np.hstack([
            np.sin(self.positional_phases + t * self.wave_speeds),
            np.zeros((self.positional_phases.shape[0], 1))
            ])
        data = waves * self.wave_amplitudes + self.wave_means
        return (data % 1, self)


class Waves(Layer):

    def __init__(self,
                physical_positions : np.array,
                wave_colors : np.array,
                wave_vectors : np.array,
                wave_speeds : np.array):
        """
        Wave colors and background: each column is a rgb colour. Average rgb across wave. Should
                     not exceed 0.5, as it's no possible to exceed 0.5 average intensity for a 
                     sine wave. After all the waves, the final column should contain the background
                     colour. The background can have any valid rgb from 0 to 1.
        Wave vectors: direction of the waves. Each column contains the xyz coordinates for
                        a vector. The length of the vectors correspond to the wavelengths.
        Wave speeds: array of frequencies, one per wave, in arbitrary units.
        """
        self.colors = wave_colors
        dot_products = np.dot(physical_positions, wave_vectors)
        wave_vector_sq_lengths = np.sum(np.square(wave_vectors), axis=0, keepdims=True)
        self.positional_phases = dot_products / wave_vector_sq_lengths
        self.wave_speeds = wave_speeds

    def get_layer(self, t: float):
        # Set to backgroun
        variable = np.hstack([
            np.sin(2 * math.pi * (self.positional_phases + t * self.wave_speeds)),
            np.zeros((self.positional_phases.shape[0], 1))
            ])
        data = np.matmul((1+variable), self.colors.transpose())
        return (data.clip(0,1), self)




class Stars(Layer):
    """Generate twinkling bright lights due to atmospheric refraction.
    
    The average color should be quite dark in order for this to work."""
    def __init__(self, size_in_pixels : int, color_rgb, 
                        fraction_of_lit_stars=0.15,
                        twinkle_interval=150,
                        twinkle_duration=0.1):
        rng = np.random.default_rng()
        self.bright_pixels_indexes = np.asarray(
                rng.random(size=(size_in_pixels,)) < fraction_of_lit_stars
            ).nonzero()[0]
        n_stars = self.bright_pixels_indexes.shape[0]
        # Lognormal distribution - arbitrary choice. Its mean is exp(0.5) = 
        star_base_brightnesses = rng.lognormal(
                        mean=math.log(max(1 / fraction_of_lit_stars, 0.0001))-0.5,
                        size=(n_stars)
                    )
        self.star_rgb_brightnesses = \
            star_base_brightnesses.reshape((n_stars,1)) * color_rgb.reshape((1, 3))# + \
            #(rng.normal(0, 0.2, size=(n_stars, 3)))

        self.size = size_in_pixels
        self.twinkle_intervals = np.random.power(2, n_stars) * twinkle_interval
        self.gaus_denom = (2 * np.power(twinkle_duration/2, 2.))

    def get_layer(self, t):
        sky = np.zeros((self.size, 3))
        modulators = 1-np.exp(
            -np.power((t % self.twinkle_intervals) - self.twinkle_intervals/2, 2.) / self.gaus_denom
        )
        sky[self.bright_pixels_indexes, :] = self.star_rgb_brightnesses * modulators.reshape((-1, 1))
        return (sky.clip(0, 1), self)


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


#-----------SETUP-----------


# Root scene controller manages the layer tree
# Custom undocumented code is just for me
class SceneController:
    def __init__(self, num_pixels, start_power_on=False, start_color_hsv=(0.8, 0.5, 0.3), scene_gamma=1):
        self.is_on = False
        self.mode = None
        self.hsv = start_color_hsv
        self.on_layer = Fill(num_pixels, hsv_to_rgb_array(*start_color_hsv))
        self.off_layer = Fill(num_pixels, np.array([0, 0, 0]))
        self.root = self.off_layer
        self.num_pixels = num_pixels
        self.scene_gamma = scene_gamma
        self.lock_event = threading.Condition()
        self._apply_new_state(start_power_on, start_color_hsv, sys.argv[1] if len(sys.argv) > 1 else "normal")
    
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

    def _make_part_scene(self, physical_positions, brightness):
        if brightness <= 0.2:
            return ScrollingRainbow(physical_positions, np.array([2, 0, 0]), 300, brightness)
        else:
            return ScrollingRainbow(physical_positions, np.array([0.5, 0, 0]), 20, brightness)

    def _apply_new_state(self, new_power_state, new_color, new_mode):
        #print("Applying", new_power_state, ",", new_color, ",", new_mode)
        if new_power_state != self.is_on or new_color != self.hsv or new_mode != self.mode:
            if not new_power_state:
                new_layer = self.off_layer
            else:
                if new_mode == "wild":
                    new_layer = WildWaves(physical_positions)
                elif new_mode == "night":
                    new_layer1 = Stars(self.num_pixels, hsv_to_rgb_array(*new_color))
                    new_layer2 = self._make_part_scene(physical_positions, new_color[-1])
                    new_layer = PlanarSplit(physical_positions, np.array([0,0,2]), np.array([0,0,1]),
                                new_layer2, new_layer1)
                elif new_mode == "basic":
                    new_layer1 = Fill(self.num_pixels, hsv_to_rgb_array(*new_color))
                    new_layer2 = self._make_part_scene(physical_positions, new_color[-1])
                    new_layer = PlanarSplit(physical_positions, np.array([0,0,2]), np.array([0,0,1]),
                                new_layer2, new_layer1)
                elif new_mode == "cinema":
                    new_layer = LinearGradient(physical_positions, 
                                            np.array([0, 4, 0]),
                                            np.array([0, -1, 0]),
                                            Fill(self.num_pixels, hsv_to_rgb_array(*new_color)),
                                            self.off_layer)
                else: #if new_mode == "normal":
                    new_layer1 = make_heavenly_waves(physical_positions, hsv_to_rgb_array(*new_color))
                    new_layer2 = self._make_part_scene(physical_positions, new_color[-1])
                    new_layer = PlanarSplit(physical_positions, np.array([0,0,2]), np.array([0,0,1]),
                                new_layer2, new_layer1)
        else: # No change
            return
        with self.lock_event:
            if "cinema" in [new_mode, self.mode]:
                self.root = GlobalFader(time.time(), 4, self.root, new_layer)
            elif self.is_on != new_power_state:
                if new_mode == "night": #start fade from bedroom
                    center = np.array([0, 2.0, 3.0])
                    speed = 2
                    soft = 1
                else:
                    center = np.array([1.5, 2.0, 3.0])
                    speed = 3
                    soft = 0.1
                self.root = RadialFader(physical_positions, time.time(), 
                                    center, speed,
                                    self.root, new_layer, soft=soft,
                                    direction_out=new_power_state)
            elif self.mode == new_mode: # just a colour change
                self.root = GlobalFader(time.time(), 1, self.root, new_layer)
            else: # fallback
                self.root = GlobalFader(time.time(), 1, self.root, new_layer)
            if new_power_state:
                self.lock_event.notify_all()

        if new_power_state:
            self.on_layer = new_layer
        self.is_on = new_power_state
        self.hsv = new_color
        self.mode = new_mode

    def wait_for_update(self, time_limit):
        """
        Wait while the output is guaranteed to be copnstant.

        Returns approximately the time waited, regardless if it was interrupted
        or not.
        """
        t0 = time.time()
        with self.lock_event:
            if self.root == self.off_layer:
                self.lock_event.wait(timeout=time_limit)
                return time.time() - t0
            return 0


    def get_data(self, t):
        with self.lock_event:
            (data, self.root) = self.root.get_layer(t)
        #if self.mask_state:
        #    data[self.mask_pixels,:] = np.array(self.hsv)
        #else:
        #    data[self.mask_pixels,:] = 0
        return data ** self.scene_gamma


def make_heavenly_waves(physical_positions, color):
    color_remain = np.array(color).reshape(3,1)
    # Make 2 waves by taking out as much as possible of the target colour
    # into the waves. Whatever is left has to become the background.
    # The waves are scaled by the Waves layer such that the average 
    # colour across the wave is equal to the given value. The wave's rgb
    # components should not exceed 0.5 because then the max output rgb 
    # value will be greater than 1.0.
    
    waves = []

    # Orange wave
    orange_factors = np.array([1, 0.5, 0])
    orange_intensity = np.min([(1-color_remain[0:2])*orange_factors[0:2],
                                        color_remain[0:2]*orange_factors[0:2]])
    orange_wave = orange_intensity*orange_factors.reshape(3, 1)
    color_remain -= orange_wave
    #print("orange", orange_intensity)
    waves.append(orange_wave)

    # Save a little light for background, so it doesn't go completely dark
    saving = np.minimum(color_remain, 10/255)
    color_remain -= saving

    # White wave component:
    # * No more than the lowest rgb component
    # * Also no more than (1-rgb) as its brightness is limited to 50% max.
    white_intensity = np.min([1-color_remain, color_remain])
    white_wave = white_intensity * np.ones((3,1))
    color_remain -= white_wave
    #print("white", white_intensity)
    waves.append(white_wave)

    # Yellow wave
    #yellow_intensity = np.min([1-color_remain[0:2], color_remain[0:2]])
    #yellow_wave = np.array((yellow_intensity, yellow_intensity, 0)).reshape(3, 1)
    #color_remain -= yellow_wave


    # Red wave
    red_intensity = np.min([1-color_remain[0], color_remain[0]])
    red_wave = np.array((red_intensity, 0, 0)).reshape(3, 1)
    color_remain -= red_wave
    #print("red", red_intensity)
    waves.append(red_wave)

    color_remain += saving

    #print("color", color, "backgrd", color_remain)

    vectors = np.array([
                            [0.9, 1.8, 0],
                            [0.9, 0.8, 0.0],
                            [0.8, 1, 0.0]
              ]).transpose()
    speeds = np.array([0.02, 0.048, 0.04])
    waves = np.hstack(waves)
    nonzero_color = np.sum(waves, axis=0) > 0
    #print(nonzero_color)
    # Background is the remaining colour
    return Waves(physical_positions,
                        wave_colors=np.hstack([waves[:,nonzero_color],color_remain]),
                        wave_vectors=vectors[:,nonzero_color],
                        wave_speeds=speeds[nonzero_color]
                )


scene = SceneController(num_pixels, start_color_hsv=(0.5, 0.5, 1.0), scene_gamma=2.1)
output_adapter = StableSpatialDithering()

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

preview_mode = (len(sys.argv) > 1)

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

    scene_data = scene.get_data(t)
    for i, node in enumerate(output_nodes):
        output_buffer = node.output_adapter.next(scene_data[node.pixel_index:node.pixel_index+node.num_pixels])
        if node.frame_skipping:
            if outputs_frame_skipping[i] == node.frame_skipping:
                outputs_frame_skipping[i] = 0
            else:
                outputs_frame_skipping[i] += 1
                continue

        packet_data = make_packet(
            output_buffer.tobytes(),
            node.universe
            )
        if not preview_mode:
            output_socket.sendto(packet_data, (node.address, node.port))

    if preview_mode:
        show_preview(output_buffer)

    dt = t - lasttime
    time.sleep(max(0, 1 / TARGET_FPS - dt))
    lasttime = t + scene.wait_for_update(STATIC_OUTPUT_SKIP_N_FRAMES / TARGET_FPS)
