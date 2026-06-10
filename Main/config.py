HOST = "127.0.0.1"
PORT = 25000
BAR_DIRECTORY = 'F:/BAR Beyond All Reason/Beyond-All-Reason/data'

# == TRAINING SETTINGS ==
# When should training occur, at each mass point or once all are reached?
TRAIN_AT_EACH_MASS_POINT = False
SHOULD_TRAIN = False # Set to False to disable training from a match (for testing the current weights or gathering data without training)

# EPISODIC TRAINING SETTINGS 
EPISODE_TIMEOUT_SECONDS = 120
END_MATCH_WHEN_ALL_MASS_REACHED = False  # if True, finalize full match/training set at 100% mass completion; if False, reset spots and continue
MASS_REACH_RADIUS = 100
MASS_FINAL_APPROACH_RADIUS = 140
MASS_DESTINATION_SWAP_THRESHOLD = 0.10
MAX_SEGMENT_STEPS = 300

# ONLINE REPLAY EXPORT SETTINGS 
# Save replay-style datasets from live agent runs and keep only the top matches by score.
SAVE_TOP_MATCH_DATASET = True
TOP_MATCHES_TO_KEEP = 15
AGENT_REPLAY_EXPORT_DIR = "Recordings/AgentReplayTop"
# Keep only a fraction of NOOP samples to better match human replay distribution.
AGENT_REPLAY_NOOP_KEEP_RATIO = 0.2

# STANDARDIZED MAP SETTINGS 
STANDARD_MAP_WIDTH = 1024
STANDARD_MAP_HEIGHT = 1024
STANDARD_MAP_Y = 256

# ACTION SETTINGS (NOW CONTINUOUS TARGETS) 
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
ESCAPE_CANDIDATE_COUNT = 3  # number of escape direction candidates to generate
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

# Building Scales
VISION_REWARD_SCALE = 0.003

# WEAPON TYPE SETTINGS 
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
FAILURE_BASE_PENALTY = 100.0
SEGMENT_TIME_PENALTY = 0.1
SEGMENT_DISTANCE_PENALTY = 0.1
SEGMENT_DAMAGE_PENALTY = 1.0
HEIGHT_DISTANCE_FACTOR = 0.2

# Number of action features for the potential field
NUM_ACTION_FEATURES = 10 # Moving
# Contains:
# 0: distance_reduction
# 1: move_magnitude
# 2: is_noop
# 3: danger_zone
# 4: enemy_distance_change
# 5: nearest_enemy_proximity
# 6: escape_alignment
# 7: skirt_alignment
# 8: dodge_viability
# 9: hazard_prediction
NUM_BUILD_FEATURES = 8 # Building 
# Contains:
# 0: friendly_proximity (Place near friendlies, away from enemies)
# 1: is_continuing_commitment (keep building/traveling to what has already started, equivalent to noop for builds)
# 2: enemy_proximity (Don't build near enemies)
# 3: mass_spot_proximity (Prefer building near mass spots)
# 4: terrain_suitability (Prefer building on flatter terrain)
# 5: blocking_proximity (Don't build if it would collide with an existing unit, provides a negative signal)
# 6: transit_progress (How far along towards a chosen building site has the unit gotten, to encourage completing building commitments once started)
# 7: prospective_vision_gain (Fraction of cells within radar radius that are unknown)

# Types of valid actions
ACTION_MOVE = 0
ACTION_BUILD = 1
NUM_DISCRETE_ACTIONS = 2

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

# ENCODER SETTINGS 
SELF_FEATURES_SIZE = 11 # Added active_build_progress, is_constructing
SELF_EMBED_SIZE = 16
ECO_FEATURES_SIZE = 2 # fEnergy, fMass
ECO_EMBED_SIZE = 8
MASS_FEATURES_SIZE = 3
MASS_EMBED_SIZE = 8
MAP_FEATURES_SIZE = 6
MAP_EMBED_SIZE = 16
VISION_EMBED_SIZE = 16
UNIT_FEATURES_SIZE = 3
ENEMY_FEATURES_SIZE = 11
FRIENDLY_EMBED_SIZE = 16
ENEMY_EMBED_SIZE = 16

# LSTM SETTINGS 
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1

# MODEL VISUALIZATION SETTINGS 
ENABLE_MODEL_GRAPH_LOG = True

# EPOCH DRIFT SETTINGS 
td_cap = 5.0

# OUTPUT WEIGHT CONSTRAINTS 
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
    "skirt_alignment",
    "dodge_viability",
    "hazard_prediction"
]

BUILD_FEATURE_NAMES = [
    "friendly_proximity",
    "is_continuing_commitment",
    "enemy_proximity",
    "mass_spot_proximity",
    "terrain_suitability",
    "blocking_proximity",
    "transit_progress",
    "prospective_vision_gain"
]

# ADAPTIVE CANDIDATE SETTINGS 
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
DISCRETE_EPSILON_MIN   = 0.05  # floor at 5% exploration permanently
DISCRETE_EPSILON_DECAY = 0.995 # decay per step
FORCE_BUILD_EVERY_N_STEPS = 50  # force ACTION_BUILD every N steps per unit

# BUILD COMMITMENT SETTINGS
BUILD_COMMIT_TIMEOUT_STEPS = 60
BUILD_TRANSIT_STALL_PENALTY = -0.5

BUILDING_REWARD_SCALE = 0.8

# SENTINEL FILE SETTINGS 
# Path to the sentinel file that signals the training script to stop.
SENTINEL_FILE_PATH = "stop.txt"
# Path to the sentinel file that contains the survival time.
TIME_FILE_PATH = "time.txt"
# Min number from TIME_FILE_PATH required for success (minute increments, running at 5x speed, so 1 = 5 real minutes).
SURVIVAL_TIME_THRESHOLD = 7

# EVAL RUN SCHEDULING 
# Run one eval match after this many training matches.
EVAL_TRAIN_RUNS_PER_CYCLE = 5
# Number of eval matches at the end of each cycle.
EVAL_RUNS_PER_CYCLE = 1
# Fixed seed used for deterministic eval matches.
EVAL_RANDOM_SEED = 1337
# Persistent counter file used to rotate between training and eval matches.
EVAL_COUNTER_FILE_PATH = "evalCheck.txt"
# Explicit run mode written by the Python server for external launchers.
RUN_MODE_FILE_PATH = "runMode.txt"
