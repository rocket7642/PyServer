HOST = "127.0.0.1"
PORT = 25000
BAR_DIRECTORY = 'F:/BAR Beyond All Reason/Beyond-All-Reason/data'

# == TRAINING SETTINGS ==
# When should training occur, at each mass point or once all are reached?
TRAIN_AT_EACH_MASS_POINT = False
SHOULD_TRAIN = False # Set to False to disable training from a match (for testing the current weights or gathering data without training)

# === EPISODIC TRAINING SETTINGS ===
EPISODE_TIMEOUT_SECONDS = 120
END_MATCH_WHEN_ALL_MASS_REACHED = False  # if True, finalize full match/training set at 100% mass completion; if False, reset spots and continue
MASS_REACH_RADIUS = 100
MASS_FINAL_APPROACH_RADIUS = 140
MASS_DESTINATION_SWAP_THRESHOLD = 0.10
MAX_SEGMENT_STEPS = 300

# === ONLINE REPLAY EXPORT SETTINGS ===
# Save replay-style datasets from live agent runs and keep only the top matches by score.
SAVE_TOP_MATCH_DATASET = True
TOP_MATCHES_TO_KEEP = 15
AGENT_REPLAY_EXPORT_DIR = "Recordings/AgentReplayTop"
# Keep only a fraction of NOOP samples to better match human replay distribution.
AGENT_REPLAY_NOOP_KEEP_RATIO = 0.2

# === STANDARDIZED MAP SETTINGS ===
STANDARD_MAP_WIDTH = 1024
STANDARD_MAP_HEIGHT = 1024
STANDARD_MAP_Y = 256

# === ACTION SETTINGS (NOW CONTINUOUS TARGETS) ===
NOOP_ACTION = "NOOP"

# Sampling settings for continuous target selection
NUM_RANDOM_TARGETS = 30
TARGET_PADDING = 50

# Command persistence settings
MIN_STEPS_BETWEEN_COMMANDS = 3
COMMAND_DISTANCE_EPS = 50
OVERRIDE_SCORE_THRESHOLD = 10.0
CANCEL_COMMAND_PENALTY = 2.0

# Potential-based move rewards
POTENTIAL_DISTANCE_SCALE = 0.1
POTENTIAL_DIRECTION_SCALE = 0.05
HEIGHT_JUMP_TOLERANCE = 15.0
HEIGHT_JUMP_PENALTY_SCALE = 0.2
IMMEDIATE_DAMAGE_PENALTY_SCALE = 2.0
PATH_DANGER_PENALTY_SCALE = 10.0
PATH_DANGER_MIN_HIT_PENALTY = 2.0

# Enemy avoidance settings
ENEMY_PROXIMITY_THRESHOLD = 400.0  # normalized distance within which enemy avoidance kicks in
ENEMY_AVOIDANCE_REWARD_SCALE = 0.15  # reward scale for increasing distance from enemies
ENEMY_RANGE_FALLOFF_BUFFER = 1.3  # multiplier on enemy weapon range for gradient falloff
ESCAPE_CANDIDATE_COUNT = 8  # number of escape direction candidates to generate
ESCAPE_CANDIDATE_DISTANCE = 150  # how far escape candidates are placed from the unit
DEFAULT_ENEMY_RANGE = 300  # fallback range if a unit has no range data
DEFAULT_UNIT_SPEED = 37.5  # fallback unit move speed if speed is unavailable
MIN_EFFECTIVE_SPEED_NORM = 1.0  # minimum normalized speed to avoid huge time estimates
DIRECT_APPROACH_SAMPLE_SPACING = 24.0  # normalized units between path samples for danger-time estimation
DIRECT_APPROACH_DPS_SECONDS_SCALE = 1  # converts DPS*seconds-in-range into score penalty
DIRECT_APPROACH_DANGER_WEIGHT_SCALE = 1.5  # amplifies penalty in high-intensity danger zones
MASS_SPOT_BLOCK_RISK_THRESHOLD = 75.0  # block a mass destination when estimated risk exceeds this value
MASS_SPOT_UNBLOCK_RISK_THRESHOLD = 30.0  # unblock only after risk drops below this lower threshold
MASS_SPOT_BLOCK_COOLDOWN_STEPS = 10  # minimum steps to keep a risky mass spot blocked

# === WEAPON TYPE SETTINGS ===
UNIT_DEFS_PATH = "data/unit_defs.json"
WEAPON_HITSCAN = 0
WEAPON_PROJECTILE = 1
WEAPON_MISSILE = 2
WEAPON_BEAM = 3
NUM_WEAPON_TYPES = 4
MAX_PROJECTILE_SPEED = 600.0  # normalization ceiling for projectile speed
MAX_AOE_RADIUS = 200.0  # normalization ceiling for AoE radius
MAX_DPS = 400.0  # normalization ceiling for DPS
DODGE_LATERAL_BONUS = 0.3  # bonus for perpendicular movement vs projectile enemies
DPS_THREAT_SCALE = 0.2  # scaling factor for DPS-weighted avoidance rewards

# Segment (mass-spot) rewards
SEGMENT_BASE_REWARD = 200.0
FAILURE_BASE_PENALTY = 150.0
SEGMENT_TIME_PENALTY = 0.5
SEGMENT_DISTANCE_PENALTY = 0.1
SEGMENT_DAMAGE_PENALTY = 1.0
HEIGHT_DISTANCE_FACTOR = 0.2

# Number of action features for the potential field
NUM_ACTION_FEATURES = 8

# Terrain sampling for path-based penalties
PATH_TERRAIN_WEIGHT = 0.25
PATH_SPIKE_THRESHOLD = 5.0
PATH_SAMPLE_COUNT = 6
TERRAIN_WAYPOINT_COUNT = 8
TERRAIN_WAYPOINT_SEARCH_RADIUS = 300

# Maximum traversable slope (height delta between adjacent cells in real space)
MAX_TRAVERSABLE_SLOPE = 1.0217

# Persistent cache for map pathfinding precomputations
MAP_CACHE_DIR = "cache/map_fields"
MAP_CACHE_VERSION = 12

# === ENCODER SETTINGS ===
SELF_FEATURES_SIZE = 9
SELF_EMBED_SIZE = 16
MASS_FEATURES_SIZE = 3
MASS_EMBED_SIZE = 8
MAP_FEATURES_SIZE = 6
MAP_EMBED_SIZE = 16
UNIT_FEATURES_SIZE = 3
ENEMY_FEATURES_SIZE = 11
FRIENDLY_EMBED_SIZE = 16
ENEMY_EMBED_SIZE = 16

# === LSTM SETTINGS ===
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1

# === MODEL VISUALIZATION SETTINGS ===
ENABLE_MODEL_GRAPH_LOG = True

# === EPOCH DRIFT SETTINGS ===
td_cap = 5.0

# === OUTPUT WEIGHT CONSTRAINTS ===
ENFORCE_DISTANCE_REDUCTION_NONNEG = True
DISTANCE_REDUCTION_INDEX = 0

FEATURE_NAMES = [
    "distance_reduction",
    #"boundary_proximity",
    "move_magnitude",
    #"terrain_steepness",
    "is_noop",
    #"height_change",
    "danger_zone",
    "enemy_distance_change",
    "nearest_enemy_proximity",
    "escape_alignment",
    "dodge_viability",
]
