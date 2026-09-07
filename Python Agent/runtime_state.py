import datetime
import time
from collections import deque

from torch.utils.tensorboard import SummaryWriter

import config

units = []
eUnits = []
eKUnits = []
eRUnits = []
fEnergy = 0
fMass = 0
# discrete_decision_counter = 0


# Map data
map_heights = None
normalized_map_heights = None
mass_spots = []
map_spots_norm = []
map_width = 0
map_height = 0
map_height_min = 0.0
map_height_max = 1.0
map_name = ""
map_info_source = ""
map_heights_source = ""
map_spots_source = ""

# Cached map embedding
cached_map_embedding = None
cached_map_embedding_device = None

# Vision image
vision_image = None
previous_vision_scores = [] # Rolling values of vision score for each unit to detect improvements/deteriorations in vision over time, which can be a useful training signal.

# Cost field for mass point pathfinding
terrain_cost_map = None
mass_cost_fields = {}

# Per-edge slope arrays (set by build_terrain_cost_map)
edge_slope_zn = None  # slope toward z-1
edge_slope_zp = None  # slope toward z+1
edge_slope_xn = None  # slope toward x-1
edge_slope_xp = None  # slope toward x+1

# Evaluation run settings
evalRun = False
run_counter = 0

# Training state variables
run_name = f"feature_based_agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(f"runs/{run_name}")
run_started_at = time.time()
step_counter = 0
train_step_counter = 0  # x-axis for Training/* scalars; step_counter freezes during end-of-match training

# Per-unit state tracking for training and analysis
previous_healths = {}
unit_max_healths = {}
previous_states = {}
previous_states_no_map = {}
previous_actions = {}
previous_discrete_actions = {} # tracks MOVE = 0, BUILD = 1
previous_positions = {}
previous_y_positions = {}
previous_distances = {}
previous_targets = {}
previous_action_scores = {}
previous_action_kinds = {}
previous_build_progress = {}
previous_command_steps = {}
cancel_command_penalties = {}
previous_forced_builds = {} # tracks whether a build action was forced due to the FORCE_BUILD_EVERY_N_STEPS rule
decision_snapshots = {} # Stores the decision snapshot for each unit at the time of action selection, which can be used for training and analysis.
# head_q_stats = {}
return_baseline = None   # EMA of Monte-Carlo returns (policy-gradient baseline)
return_var = 1.0         # EMA variance of returns (advantage normalization)
return_ema_alpha = 0.01

# Build commitment and progress tracking for units
build_committed_target = {}
build_committed_since_step = {}
build_committed_distance = {}
build_released_step = {}
danger_released_sites = {}

# Vision and target tracking
previous_build_vision_baseline = 0
previous_chosen_targets = {}

# LSTM hidden states for each unit, used to maintain temporal context across steps in the agent's decision-making process.
lstm_hidden_states = {}
previous_lstm_hidden_states = {}

# Mass spot tracking for pathfinding and resource gathering
mass_destinations = {}
mass_destination_distances = {}
mass_spot_blocked_until = {}

# Per-unit adaptive candidate templates used to generate context-relative movement variants.
adaptive_candidate_templates = {}
adaptive_template_next_id = 1

# Per-unit adaptive build templates used to generate context-relative build variants.
last_build_step = {}

model_graph_logged = False

# Tracking of mass spots visited during the game
visited_mass_spots = set()
visited_mass_spots_norm = set()
mass_cycle_completions = 0

# Tracking of consecutive inactive steps for each unit, which can be used to detect units that are stuck or not contributing to the game.
consecutive_inactive = {}
last_mass_visit = {}

# Tracking of match segments and statistics for analysis and training purposes.
segment_buffers = {}
segment_stats = {}
match_buffer = []
match_segment_summaries = []
current_match_samples = []

# Match finalization guards
match_finalized = False
forced_terminal_success = False

# pause time for mid match training (keep in mind this is currently not used)
pause_time = 5
process_times = [5]

# Unit definition data loaded from JSON at startup
unit_defs_loaded = False

# Sentinel Time path for tracking survival time in the training script
sentinel_time_path = None

# EMA tracking for Q-values to stabilize training and provide a smoothed estimate of expected returns.
move_q_ema = 0.0
build_q_ema = 0.0
q_ema_alpha = 0.01