HOST = "127.0.0.1"
PORT = 25000
BAR_DIRECTORY = 'F:/BAR Beyond All Reason/New BAR/Beyond-All-Reason/data'

# Types of valid actions
ACTION_MOVE = 0
ACTION_BUILD = 1
NUM_DISCRETE_ACTIONS = 2

# Training Settings
# When should training occur, at each mass point or once all are reached?
TRAIN_AT_EACH_MASS_POINT = False
SHOULD_TRAIN = False  # Set to False to disable training from a match (for testing the current weights or gathering data without training)

# Action Override — None for normal operation;
# ACTION_MOVE (0) or ACTION_BUILD (1) to pin the head for scenario capture.
FORCE_DISCRETE_ACTION = None

# episode timeout and mass completion settings 
EPISODE_TIMEOUT_SECONDS = 120
END_MATCH_WHEN_ALL_MASS_REACHED = False  # if True, finalize full match/training set at 100% mass completion; if False, reset spots and continue
MASS_REACH_RADIUS = 100
MASS_FINAL_APPROACH_RADIUS = 140
MASS_DESTINATION_SWAP_THRESHOLD = 0.10
MAX_SEGMENT_STEPS = 300

# Replay export settings for training data collection
# Save replay-style datasets from live agent runs and keep only the top matches by score.
SAVE_TOP_MATCH_DATASET = True
TOP_MATCHES_TO_KEEP = 15
AGENT_REPLAY_EXPORT_DIR = "Recordings/AgentReplayTop"
# Keep only a fraction of NOOP samples to better match human replay distribution.
AGENT_REPLAY_NOOP_KEEP_RATIO = 0.2

# Default map dimensions for normalization of coordinates and distances
STANDARD_MAP_WIDTH = 1024
STANDARD_MAP_HEIGHT = 1024
STANDARD_MAP_Y = 256

# Noop definition for action selection
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
ENEMY_PROXIMITY_THRESHOLD = 75.0  # normalized ≈ 600 world ≈ 2x typical weapon range
ENEMY_AVOIDANCE_REWARD_SCALE = 0.15  # reward scale for increasing distance from enemies
ENEMY_RANGE_FALLOFF_BUFFER = 1.3  # multiplier on enemy weapon range for gradient falloff
ESCAPE_CANDIDATE_COUNT = 3  # number of escape direction candidates to generate
ESCAPE_CANDIDATE_DISTANCE = 150  # how far escape candidates are placed from the unit
DEFAULT_ENEMY_RANGE = 300  # fallback range if a unit has no range data
DEFAULT_UNIT_SPEED = 37.5  # fallback unit move speed if speed is unavailable
MIN_EFFECTIVE_SPEED_NORM = 1.0  # minimum normalized speed to avoid huge time estimates
DIRECT_APPROACH_SAMPLE_SPACING = 24.0  # normalized units between path samples for danger-time estimation
DIRECT_APPROACH_DPS_SECONDS_SCALE = 1  # converts DPS*seconds-in-range into score penalty
DIRECT_APPROACH_DANGER_WEIGHT_SCALE = 1.5  # amplifies penalty in high-intensity danger zones
MASS_SPOT_BLOCK_RISK_THRESHOLD = 500.0  # block a mass destination when estimated risk exceeds this value
MASS_SPOT_UNBLOCK_RISK_THRESHOLD = 100.0  # unblock only after risk drops below this lower threshold
MASS_SPOT_BLOCK_COOLDOWN_STEPS = 10  # minimum steps to keep a risky mass spot blocked

# Building Scales
VISION_REWARD_SCALE = 0.003

# Weapon types and normalization ceilings for avoidance scoring 
UNIT_DEFS_PATH = "data/unit_defs.json"
WEAPON_HITSCAN = 0
WEAPON_PROJECTILE = 1
WEAPON_MISSILE = 2
WEAPON_BEAM = 3
NUM_WEAPON_TYPES = 4
MAX_PROJECTILE_SPEED = 600.0  # normalization ceiling for projectile speed (only things faster are lasers, but they are instant)
# Both maxes below are utilized in my scenario testing for normalization, they do not cover all values possible in game
MAX_AOE_RADIUS = 200.0  # normalization ceiling for AoE radius (Change this if you add new units with higher AoE than the current max)
MAX_DPS = 400.0  # normalization ceiling for DPS (Change this is you add new units with higher DPS than the current max)
DODGE_LATERAL_BONUS = 0.3  # bonus for perpendicular movement vs projectile enemies
DPS_THREAT_SCALE = 0.2  # scaling factor for DPS-weighted avoidance rewards

# Noop danger scale: Make standing still in danger still count as risky compared to moving
NOOP_DANGER_SCALE = 500

# Segment (mass-spot) rewards
SEGMENT_BASE_REWARD = 200.0
FAILURE_BASE_PENALTY = 100.0
SEGMENT_TIME_PENALTY = 0.1
SEGMENT_DISTANCE_PENALTY = 0.1
SEGMENT_DAMAGE_PENALTY = 1.0
HEIGHT_DISTANCE_FACTOR = 0.2

# Number of action features for the potential field
NUM_ACTION_FEATURES = 10 # Moving
NUM_BUILD_FEATURES = 8 # Building 

FEATURE_NAMES = [
    "distance_reduction", # (How much closer to the target the move would get the unit)
    #"boundary_proximity",
    "move_magnitude", # (How far the move would take the unit)
    #"terrain_steepness",
    "is_noop", # (Whether the move is a NOOP, which is generally bad in dangerous situations)
    #"height_change",
    "danger_zone", # (Whether the move would place the unit in a danger zone)
    "enemy_distance_change", # (Whether the move would increase or decrease distance to the nearest enemy)
    "nearest_enemy_proximity", # (How close the nearest enemy is to the unit)
    "escape_alignment", # (Whether the move is aligned with an escape vector away from enemies)
    "skirt_alignment", # (Whether the move is aligned with a skirt vector around enemies)
    "dodge_viability", # (Whether the move is aligned with a dodge vector, based on enemy weapon types and ranges)
    "hazard_prediction" # (Predicted hazard level of the move)
]

BUILD_FEATURE_NAMES = [
    "friendly_proximity", # (Place near friendlies, away from enemies)
    "is_continuing_commitment", # (Keep building/traveling to what has already started, equivalent to noop for builds)
    "enemy_proximity", # (Don't build near enemies)
    "mass_spot_proximity", # (Prefer building near mass spots)
    "terrain_suitability", # (Prefer building on flatter terrain)
    "blocking_proximity", # (Don't build if it would collide with an existing unit, provides a negative signal)
    "transit_progress", # (How far along towards a chosen building site has the unit gotten, to encourage completing building commitments once started)
    "prospective_vision_gain" # (Fraction of cells within radar radius that are unknown)
]

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

# Mass value weighting for cost fields: higher pulls path cost down near high-value spots
MASS_VALUE_ALPHA = 0.5  # in [0,1]

# Embedding sizes for various feature groups 
SELF_FEATURES_SIZE = 11 # Added active_build_progress, is_constructing
SELF_EMBED_SIZE = 16
ECO_FEATURES_SIZE = 2 # fEnergy, fMass
ECO_EMBED_SIZE = 8
MASS_FEATURES_SIZE = 3
MASS_EMBED_SIZE = 8
MAP_FEATURES_SIZE = 6
MAP_EMBED_SIZE = 16
VISION_EMBED_SIZE = 16
UNIT_FEATURES_SIZE = 7 # Was 3, 
ENEMY_FEATURES_SIZE = 11
FRIENDLY_EMBED_SIZE = 16
ENEMY_EMBED_SIZE = 16

# LSTM settings for temporal context in the agent's decision-making 
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1

# Model graph logging for TensorBoard visualization
# Not actually used anymore, but leaving the option in case we want to allow tensorboard logging to be toggled later
ENABLE_MODEL_GRAPH_LOG = True

# Epoch drift cap
# Shifted away from td_cap to allow faster training as the project came to a close, but maintained if reversion is worth
# Reference commit: "Build improvements hopefully, aiming to unstuck the head" for what was modified at that time
# td_cap = 5.0

# tune this; higher = stronger resistance to collapse
entropy_coeff = 0.15 # Temporary raised from 0.05 while builds are failing

 
# Output weight sign constraints — enforced via softplus in agent_core.constrain_head_weights
MOVE_NONPOS_INDICES = [1]           # move_magnitude
MOVE_NONNEG_INDICES = [0, 7, 8]     # distance_reduction, skirt_alignment, dodge_viability
BUILD_NONNEG_INDICES = [1, 6, 7]    # is_continuing, transit_progress, prospective_vision_gain
BUILD_NONPOS_INDICES = [2, 5]       # enemy_proximity, blocking_proximity



# Adaptive candidate generation settings 
# Enables context-anchored candidate templates that mutate around base candidates.
ADAPTIVE_CANDIDATES_ENABLED = True
# Max persistent templates tracked per unit.
ADAPTIVE_MAX_TEMPLATES_PER_UNIT = 24
# Mutated adaptive candidates generated per decision step.
ADAPTIVE_MUTATIONS_PER_STEP = 8
# Mutation noise applied to template polar offsets.
ADAPTIVE_MUTATION_DISTANCE_STD = 25.0
ADAPTIVE_MUTATION_ANGLE_STD = 0.35
# Lower bounds reached as training progresses.
ADAPTIVE_MUTATIONS_PER_STEP_MIN = 2
ADAPTIVE_MUTATION_DISTANCE_STD_MIN = 6.0
ADAPTIVE_MUTATION_ANGLE_STD_MIN = 0.08
# Exponential decay schedule: value(t) = min + (start - min) * exp(-rate * t)
# Decay progress source can be "step" (runtime steps) or "epoch" (mass-cycle completions).
ADAPTIVE_MUTATION_DECAY_SOURCE = "step"
ADAPTIVE_MUTATION_DECAY_RATE = 0.0002
# Candidate acceptance and lifecycle controls.
ADAPTIVE_MIN_CANDIDATE_DISTANCE = 8.0
ADAPTIVE_MAX_CANDIDATE_DISTANCE = 80.0
ADAPTIVE_SCORE_DECAY = 0.97
ADAPTIVE_PROMOTION_SCORE = 0.05
ADAPTIVE_PRUNE_SCORE = -0.5
ADAPTIVE_PRUNE_MIN_VISITS = 4

# Decay for head selection logits to encourage exploration of different heads.
DISCRETE_EPSILON_START = 0.3   # 30% random discrete action at start
DISCRETE_EPSILON_MIN   = 0.15  # floor at 5% exploration permanently # Temporary raised from 0.05 while builds are failing
DISCRETE_EPSILON_DECAY = 0.995 # decay per step
FORCE_BUILD_EVERY_N_STEPS = 50  # force ACTION_BUILD every N steps per unit

# Build commitment and progress timeout settings
BUILD_COMMIT_TIMEOUT_STEPS = 15
BUILD_TRANSIT_STALL_PENALTY = -0.5
# FORCED_BUILD_EXPLORE_WEIGHT = 0.1
BUILD_RESEND_COOLDOWN_STEPS = 5
POST_BUILD_DECISION_WINDOW = 3
DANGER_SITE_COOLDOWN_STEPS = 30

# Build reward scaling for training signal
BUILDING_REWARD_SCALE = 0.8
BUILD_PROGRESS_REWARD_SCALE = 10.0
BUILD_CANDIDATE_RADIUS = 3
# BUILD_TARGET_SWITCH_MARGIN = 0.15
# BUILD_CONTINUITY_BONUS = 0.75

# Sentinel file for automatic training termination and survival time tracking
# Path to the sentinel file that signals the training script to stop.
SENTINEL_FILE_PATH = "stop.txt"
# Path to the sentinel file that contains the survival time.
TIME_FILE_PATH = "time.txt"
# Min number from TIME_FILE_PATH required for success (minute increments, running at 5x speed, so 1 = 5 real minutes).
# If you run on a different multiplyer, adjust this value
SURVIVAL_TIME_THRESHOLD = 7

# Evaluation settings for training cycles 
# Run one eval match after this many training matches.
EVAL_TRAIN_RUNS_PER_CYCLE = 5
# Number of eval matches at the end of each cycle.
EVAL_RUNS_PER_CYCLE = 1
# Fixed seed used for deterministic eval matches.
EVAL_RANDOM_SEED = 1337 # Leet
# Persistent counter file used to rotate between training and eval matches.
EVAL_COUNTER_FILE_PATH = "evalCheck.txt"
# Explicit run mode written by the Python server for external launchers.
RUN_MODE_FILE_PATH = "runMode.txt"
