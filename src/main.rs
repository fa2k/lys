extern crate paho_mqtt as mqtt;

use std::{
    net::UdpSocket,
    process,
    time::Duration,
    net::SocketAddr,
    cmp,
    fmt
};
use binary_layout::prelude::*;
use rand::Rng;
use gethostname;

define_layout!(artnet_packet, LittleEndian, {
    tag: [u8; 8],
    opcode: u16,      // 0x5000
    version_high: u8, // 0
    version_low: u8,  // 14
    sequence: u8,
    physical: u8,
    universe: u16,
    length_high: u8,
    length_low: u8,
    data: [u8]
});

const HEADER_SIZE: usize = 18; // Maybe binary_layout could help?

struct Node {
    color_topic: String,
    address: SocketAddr,
    start_universe: u16,
    num_pixels: u16,
    fps: u16,
}

struct HsbColor {
    h: f64,
    s: f64,
    b: f64
}

struct RgbColor {
     r: f64, g: f64, b: f64
}

impl RgbColor {
    fn comps(& self) -> [f64; 3] {
        [self.r, self.g, self.b]
    }
    fn from_comps(data: [f64; 3]) -> RgbColor {
        RgbColor{
            r: data[0],
            g: data[1],
            b: data[2]
        }
    }
}

impl fmt::Display for RgbColor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RgbColor(r: {}, g: {}, b: {})", self.r, self.g, self.b)
    }
}

const BLACK: HsbColor = HsbColor {h: 0.0, s: 0.0, b: 0.0};

struct State {
    on_color: HsbColor
}

fn main() {

    // Setup MQTT input
    let host = "tcp://192.168.1.8:1883";
    let create_opts = mqtt::CreateOptionsBuilder::new()
        .server_uri(host)
        .client_id(gethostname::gethostname().into_string().unwrap() + &"_m2a".to_string())
        .finalize();

    let mut cli = mqtt::Client::new(create_opts).unwrap_or_else(|err| {
                println!("Error creating the client: {:?}", err);
                process::exit(1);
            });

    let rx = cli.start_consuming();

    
    let conn_opts = mqtt::ConnectOptionsBuilder::new()
        .automatic_reconnect(Duration::from_secs(5), Duration::from_secs(600))
        .finalize();

    // Connect and wait for it to complete or fail.
    if let Err(e) = cli.connect(conn_opts) {
         println!("Unable to connect:\n\t{:?}", e);
         process::exit(1);
    }

    let nodes = [
        Node{
            color_topic: "stue/artnet/tak/farge".to_string(),
            address: "192.168.1.224:6454".parse().unwrap(),
            start_universe: 1,
            num_pixels: 895,
            fps: 100
        },
        Node{
            color_topic: "stue/artnet/bakHyller/farge".to_string(),
            address: "192.168.1.226:6454".parse().unwrap(),
            start_universe: 0,
            num_pixels: 3,
            fps: 30
        },
        Node{
            color_topic: "stue/artnet/barskap/farge".to_string(),
            address: "192.168.1.226:6454".parse().unwrap(),
            start_universe: 1,
            num_pixels: 1,
            fps: 30
        },
    ];

    let topics: Vec<&String> = nodes.iter().map( |node| &node.color_topic ).collect();
    
    let zeros: Vec<i32> =  vec![0; topics.len()];
    if let Err(e) = cli.subscribe_many(&topics, &zeros) {
        println!("Error encountered while subscribing to topics: {:?}", e);
        process::exit(1);
    }


    let mut states: Vec<State> = nodes.iter()
                    .map( |_|  State { on_color: BLACK } )
                    .collect();

    let socket = UdpSocket::bind("0.0.0.0:0").expect("Unable to bind UDP client socket");
    for msg in rx.iter() {
        if let Some(msg) = msg {
            for i in 0..nodes.len() {
                let node = &nodes[i];
                if msg.topic() == node.color_topic {
                    let data = get_channels_for_command(
                        &mut states[i],
                        node.num_pixels,
                        &msg.payload_str()
                    );
                    send_data(&socket, node, &data);
                }
            }
        }
        else {
            println!("lys warning: They hung up. Will retry.")
        }
    }

    if cli.is_connected() {
        cli.unsubscribe_many(&topics).unwrap();
        cli.disconnect(None).unwrap();
    }
}

fn hsv2rgb(hsv: &HsbColor) -> RgbColor
{
    //https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
    if(hsv.s <= 0.0) {       // < is bogus, just shuts up warnings
        return RgbColor{
            r: hsv.b,
            g: hsv.b,
            b: hsv.b
        }
    }
    let hh = if hsv.h >= 360.0 { 0.0 } else { hsv.h } / 60.0;
    let ff = hh - hh.floor();
    let p = hsv.b * (1.0 - hsv.s);
    let q = hsv.b * (1.0 - (hsv.s * ff));
    let t = hsv.b * (1.0 - (hsv.s * (1.0 - ff)));

    match (hh as u64) {
        0 =>
            RgbColor{
                r: hsv.b,
                g: t,
                b: p
            },
        1 =>
            RgbColor {
                r: q,
                g: hsv.b,
                b: p
            },
        2 =>
            RgbColor{
                r: p,
                g: hsv.b,
                b: t
            },
        3 =>
            RgbColor{
                r: p,
                g: q,
                b: hsv.b
            },
        4 =>
            RgbColor {
                r: t,
                g: p,
                b: hsv.b
            },
        _ =>
            RgbColor {
                r: hsv.b,
                g: p,
                b: q
            }
    }
    
    
    /*/let mut H: f64 = hsv.h.into();
    let S: f64 = hsv.s.into();
    let V: f64 = hsv.b.into();

    if H == 360.0 {
        H = 0.0;
    }
    else {
        H = H / 60.0;
    }
    let fract = H - H.floor();

    let P = V*(1. - S);
    let Q = V*(1. - S*fract);
    let T = V*(1. - S*(1. - fract));

    if      0. <= H && H < 1. {
        return RgbColor{r: V, g: T, b: P};
    }
    else if 1. <= H && H < 2. {
        return RgbColor{r: Q, g: V, b: P};
    }
    else if 2. <= H && H < 3. {
        return RgbColor{r: P, g: V, b: T};
    }
    else if 3. <= H && H < 4. {
        return RgbColor{r: P, g: Q, b: V};
    }
    else if 4. <= H && H < 5. {
        return RgbColor{r: T, g: P, b: V};
    }
    else if 5. <= H && H < 6. {
        return RgbColor{r: V, g: P, b: Q};
    }
    else {
        return RgbColor{r: 0.0, g: 0.0, b: 0.0};
    }*/
}

fn gamma_convert(rgb: RgbColor) -> RgbColor {
    RgbColor::from_comps(
        rgb.comps().map(
            |c| c.powf(2.1)
        )
    )
}


fn get_channels_dither(col: &HsbColor, npixels: u16) -> Vec<u8> {
    let rgb = gamma_convert(hsv2rgb(col));
    let mut data = Vec::with_capacity((npixels * 3).into());
    
    //println!("{}", rgb);

    let mut intparts = [0u8; 3];
    let mut fracparts = [0.0f64; 3];

    for ic in [0, 1, 2] {
        let value = (rgb.comps()[ic] * 255.0).clamp(0.0, 255.0);
        intparts[ic] = value as u8;
        fracparts[ic] = value - intparts[ic] as f64;
    }
    let mut rng = rand::thread_rng();
    for _i in 0..npixels {
        for ic in [0, 1, 2] {
            data.push(
                intparts[ic] + if rng.gen_range(0.0..1.0) < fracparts[ic] { 1 } else { 0 }
            );
        }
    }
    data
}

fn string_to_color(text: &str) -> HsbColor {
    let parts: Vec<&str> = text.split(",").collect();
    if parts.len() == 3 {
        HsbColor {
            h: parts[0].parse().unwrap_or(0.0),         // Degrees
            s: parts[1].parse().unwrap_or(0.0) / 100.0, // Percent
            b: parts[2].parse().unwrap_or(0.0) / 100.0, // Percent
        }
    }
    else {
        HsbColor{ h: 0.0, s: 0.0, b: 0.0 }
    }
}

fn get_channels_for_command(state: &mut State, npixels: u16, msg: &str)
                         -> Vec<u8> {
    match msg {
        "ON" =>  get_channels_dither(&state.on_color, npixels),
        "OFF" => get_channels_dither(&BLACK, npixels),
        _ =>     {
            state.on_color = string_to_color(msg);
            get_channels_dither(&state.on_color, npixels)
        }
    }
}

fn send_data(socket: &UdpSocket, node: &Node, data: &[u8]) {
    let n_packet: usize = ((node.num_pixels + 169) / 170).into();
    for i in 0..n_packet {
        let buf_end = cmp::min(data.len(), (i+1)*510);
        let data_channels_buffer = &data[(i*510)..buf_end];
        let packet_size = HEADER_SIZE + data_channels_buffer.len();
        let mut packet_data = vec![0; packet_size];
        packet_data[0..7].clone_from_slice("Art-Net".as_bytes());
        let mut view = artnet_packet::View::new(packet_data);
        view.opcode_mut().write(0x5000);
        view.version_low_mut().write(14);
        view.universe_mut().write(node.start_universe + i as u16);
        view.length_high_mut().write((data_channels_buffer.len() >> 8) as u8);
        view.length_low_mut().write(data_channels_buffer.len() as u8);
        view.data_mut().data_mut().clone_from_slice(data_channels_buffer);
        let ready_packet_data = view.into_storage();
        socket.send_to(&ready_packet_data, node.address).expect("Send packet error");
    }
}

//struct Base;
//impl layer::Layer for Base {
//    
//fn getparams() -> std::vec::Vec<layer::ParameterSpec> { todo!() }
//fn getpar(_: u8) -> f64 { todo!() }
//fn setpar(_: u8, _: f64) { todo!() }
//fn makeframe() { todo!() }
//}
