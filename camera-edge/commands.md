# Camera Edge Runtime Commands

## MediaMTX On The Receiver / PC

Windows:

```bash
./camera-edge/mediamtx.exe camera-edge/mediamtx.yml
```

Linux:

```bash
./camera-edge/mediamtx camera-edge/mediamtx.yml
```

## Sender On Jetson Nano

Edit `JETSON_SENDER_CONFIG` in `antropi_config.py`, then run:

```bash
python camera-edge/edge_rgb_depth_sender.py
```

Useful sender fields in `antropi_config.py`:

```bash
JETSON_SENDER_CONFIG.profile_name
JETSON_SENDER_CONFIG.run_duration_sec
JETSON_SENDER_CONFIG.print_config_only
```

Supported production profiles:

```bash
wired_lz4
wireless_lz4
wired_zstd
wireless_zstd
```

## Main Application On The Server / PC

Edit `JETSON_PICK_PIPELINE_CONFIG` in `antropi_config.py`, then run:

```bash
python main.py
```

Useful PC-side fields in `antropi_config.py`:

```bash
JETSON_PICK_PIPELINE_CONFIG.profile_name
JETSON_PICK_PIPELINE_CONFIG.rtsp_url_override
JETSON_PICK_PIPELINE_CONFIG.rtsp_transport_override
JETSON_PICK_PIPELINE_CONFIG.depth_bind_ip_override
JETSON_PICK_PIPELINE_CONFIG.depth_port_override
JETSON_PICK_PIPELINE_CONFIG.startup_timeout_s
JETSON_PICK_PIPELINE_CONFIG.sam_prompt
JETSON_PICK_PIPELINE_CONFIG.sam_valid_frames
JETSON_PICK_PIPELINE_CONFIG.sam_orient_median_repeats
JETSON_PICK_PIPELINE_CONFIG.sam_min_points
JETSON_PICK_PIPELINE_CONFIG.sam3_multi_instance
JETSON_PICK_PIPELINE_CONFIG.verbose
```

If `profile_name` is `None`, both entrypoints follow `CAMERA_EDGE_RUNTIME_CONFIG.active_profile_name` from `antropi_config.py`.
