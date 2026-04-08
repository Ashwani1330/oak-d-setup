# antropi_config.py
import numpy as np

PURPLE_RGB = (132, 71, 255)
PURPLE_BGR = (255, 71, 132)

# Robot
TOOL_LENGTH_OFFSET = 180
HOME_CHECKPOINT = {"x": 450, 
                   "y": 0, 
                   "z": 450, 
                    "a": 180, 
                    "b": 0, 
                    "c": 180, 
                    "s": int("010", 2), 
                    "t": int("001010", 2)
}

PICK_CHECKPOINT = {"x": 505.82, 
                   "y": -89.82, 
                   "z": 2, 
                    "a":  155.85,
                    "b": -10.67, 
                    "c": 180,
                    "s": -1, 
                    "t": -1
}


# T_FLANGE_CAM = np.array(
#     [[ 0.00806,  -0.999797,  0.018449, -0.070823],
#      [ 0.999967,  0.008071,  0.000534,  0.],
#      [-0.000683,  0.018444,  0.99983,   0.021132],
#      [ 0.,        0.,        0.,        1.      ]],
#      dtype=np.float64,
# ) # tested with charuco


T_FLANGE_CAM = np.array(
    [[0.00806, -0.999797, 0.018449, -0.070823],
     [0.999967, 0.008071, 0.000534, 0.0],
     [-0.000683, 0.018444, 0.99983, 0.055],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)  # testing with sam3 for z


# T_FLANGE_CAM = np.array(
#     [[ 0.000276, -0.999735,  0.023022, -0.070588],
#      [ 1.00,      0.000262, -0.000609,  0.002676],
#      [ 0.000603,  0.023022,  0.999735,  0.03698 ],
#      [ 0.,        0.,        0.,        1.      ]],
#     dtype = np.float64,
# )

T_FLANGE_TOOL = np.array(
    [[1.0, 0.0, 0.0, 1.38 / 1000],
     [0.0, 1.0, 0.0, 0.04 / 1000],
     [0.0, 0.0, 1.0, 255.02 / 1000],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
T_TOOL_FLANGE = np.linalg.inv(T_FLANGE_TOOL)


# Gantry
# MQTT_BROKER_IP = "192.168.1.101"
MQTT_BROKER_IP = "192.168.1.183"
MQTT_BROKER_PORT = 1883

# Sam2
SAM_CHECKPOINT = r"C:\Users\wingway\Documents\segment-anything-2-real-time-main\checkpoints\sam2.1_hiera_small.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


# Gripper (CH343)
GRIPPER_COM_PORT = 'COM4'
GRIPPER_OPEN_MIN_MM = 50.0
GRIPPER_CLOSE_MM = 60.0
GRIPPER_PREOPEN_EXTRA_MM = 30.0
GRIPPER_PADDING_MM = 17.0
GRIPPER_OPEN_MAX_MM = 95.0

SERVER_PC_IP = "172.31.1.148"
KUKA_ROBOT_IP = "172.31.1.147"
KUKA_CMD_PORT = 16000
    
CURRENT_ROBOT_POS_IP = "127.0.0.1"
CURRENT_ROBOT_POS_PORT = 16001

# =========================
# Robot Bridge Config
# =========================

from dataclasses import dataclass

@dataclass
class RobotBridgeConfig:
    # Robot telegram (mxAutomation) network
    pc_ip: str = SERVER_PC_IP          # "172.31.1.148"
    robot_ip: str = KUKA_ROBOT_IP      # "172.31.1.147"
    axis_group: int = 1
    rx_port: int = 1337  # robot -> PC
    tx_port: int = 1336  # PC -> robot
    cycle_ms: int = 20

    # Command UDP (external controller -> bridge)
    cmd_bind_ip: str = "0.0.0.0"
    cmd_port: int = KUKA_CMD_PORT      # 16000 (your external command port)

    # Status UDP (bridge -> external controller)
    status_dst_ip: str = CURRENT_ROBOT_POS_IP     # "127.0.0.1"
    status_dst_port: int = CURRENT_ROBOT_POS_PORT # 16001
    status_hz: float = 20.0  # 10–30Hz typical

    # Motion params
    target_tool: int = 5
    target_base: int = 0
    override_pct: int = 10
    vel_pct: int = 10
    acc_pct: int = 20
    buffermode: int = 1  # 1=ABORTING, 2=BUFFERED

    # Home target (mm/deg)
    home_x: float = 450.0
    home_y: float = 0.0
    home_z: float = 650.0
    home_a: float = -180.0
    home_b: float = 0.0
    home_c: float = 180.0
    home_status: int = -1
    home_turn: int = -1

    # Defaults for incoming MOVE_ABS if status/turn not provided
    default_status: int = -1
    default_turn: int = -1

    # Timeouts
    ready_timeout_s: float = 20.0
    autostart_timeout_s: float = 20.0
    move_timeout_s: float = 120.0

    # Command freshness / safety
    cmd_stale_s: float = 2.0          # ignore move commands older than this
    max_cmds_per_cycle: int = 10      # avoid spending too long draining UDP


@dataclass(frozen=True)
class SamRunnerConfig:
    prompt: str = "black cylinder pipe piece"
    valid_frames: int = 20
    min_points: int = 80
    orient_median_repeats: int = 5
    voxel_scene_mm: float = 8.0
    voxel_obj_mm: float = 2.0
    max_viz_points: int = 300000
    tint_rgb_u8: tuple[int, int, int] = (80, 220, 120)
    tint_alpha: float = 0.3
    side_dot_abs_max: float = 0.3
    refine_axis_using_side: bool = True
    axis_refine_iters: int = 2
    min_side_points: int = 200
    refine_axis_from_normals: bool = True
    normal_radius_mult: float = 5.0
    normal_max_nn: int = 30

    def validate(self) -> None:
        """Validate runtime capture settings used by the SAM runner session."""
        if int(self.valid_frames) <= 0:
            raise ValueError("valid_frames must be > 0")
        if int(self.min_points) < 3:
            raise ValueError("min_points must be >= 3")
        if int(self.orient_median_repeats) <= 0:
            raise ValueError("orient_median_repeats must be > 0")


DEFAULT_SAM_RUNNER_CONFIG = SamRunnerConfig()


@dataclass
class CameraEdgeRuntimeConfig:
    active_profile_name: str = "wired_zstd"
    server_pc_wifi_ip: str = MQTT_BROKER_IP
    server_pc_wired_ip: str = SERVER_PC_IP
    orin_wifi_ip: str = "192.168.1.211"
    orin_wired_ip: str = "172.31.1.211"
    server_pc_wifi_nic: str = "Wi-Fi"
    server_pc_wired_nic: str = "Ethernet"
    orin_wifi_nic: str = "wlP1p1s0"
    orin_wired_nic: str = "enP8p1s0"


@dataclass
class JetsonSenderConfig:
    profile_name: str | None = None
    run_duration_sec: float | None = None
    print_config_only: bool = False


@dataclass
class JetsonPickPipelineConfig:
    profile_name: str | None = None
    rtsp_url_override: str | None = None
    rtsp_transport_override: str | None = None
    depth_bind_ip_override: str | None = None
    depth_port_override: int | None = None
    startup_timeout_s: float = 0.0
    stale_depth_ms_override: int | None = None
    socket_rcvbuf_override: int | None = None
    max_rgb_buffer_override: int | None = None
    max_rgb_depth_delta_ms_override: float | None = None
    min_mm_override: int | None = None
    max_mm_override: int | None = None
    sam_prompt: str = DEFAULT_SAM_RUNNER_CONFIG.prompt
    sam_valid_frames: int = DEFAULT_SAM_RUNNER_CONFIG.valid_frames
    sam_orient_median_repeats: int = DEFAULT_SAM_RUNNER_CONFIG.orient_median_repeats
    sam_min_points: int = DEFAULT_SAM_RUNNER_CONFIG.min_points
    sam3_multi_instance: bool = True
    verbose: bool = True


JETSON_SENDER_CONFIG = JetsonSenderConfig()

JETSON_PICK_PIPELINE_CONFIG = JetsonPickPipelineConfig()
CAMERA_EDGE_RUNTIME_CONFIG = CameraEdgeRuntimeConfig()
