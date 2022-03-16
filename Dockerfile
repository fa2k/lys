FROM rust:1.58 as build
# Install cmake for paho-mqtt
RUN apt-get update && apt-get install -y cmake
WORKDIR /lys
COPY Cargo.toml .
COPY src src
RUN cargo build --release

FROM alpine:latest
COPY --from=build /target/release/lys .
CMD ["/lys"]
