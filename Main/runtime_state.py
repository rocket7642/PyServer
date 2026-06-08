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

evalRun = False
run_counter = 0

run_name = f"feature_based_agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(f"runs/{run_name}")
run_started_at = time.time()
step_counter = 0

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
previous_command_steps = {}
cancel_command_penalties = {}

build_committed_target = {}
build_committed_since_step = {}
build_committed_distance = {}

previous_chosen_targets = {}

lstm_hidden_states = {}
previous_lstm_hidden_states = {}

mass_destinations = {}
mass_destination_distances = {}
mass_spot_blocked_until = {}

# Per-unit adaptive candidate templates used to generate context-relative movement variants.
adaptive_candidate_templates = {}
adaptive_template_next_id = 1

last_build_step = {}

model_graph_logged = False

visited_mass_spots = set()
visited_mass_spots_norm = set()
mass_cycle_completions = 0

consecutive_inactive = {}
last_mass_visit = {}

segment_buffers = {}
segment_stats = {}
match_buffer = []
match_segment_summaries = []
current_match_samples = []

# Match finalization guards
match_finalized = False
forced_terminal_success = False

pause_time = 5
process_times = [5]

# Unit definition data loaded from JSON at startup
unit_defs_loaded = False

# Sentinel Time path for tracking survival time in the training script
sentinel_time_path = None