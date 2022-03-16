# lys

Sorry, this Rust program and the docker image is useless for anyone else.
It has hardcoded IP addresses (TODO: fix).

Convert from MQTT to Art-Net protocol.

Currently very simple, sends the same colour to an array of RGB pixels.



Known issue: doesn't always send all packets on Windows / BSD (ARP resolution).
