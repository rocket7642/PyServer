import datetime
from collections import deque

from torch.utils.tensorboard import SummaryWriter

import config

units = []
eUnits = []
eKUnits = []

# Map data
map_heights = None
normalized_map_heights = None
mass_spots = []
map_spots_norm = []
map_width = 0
map_height = 0
map_height_min = 0.0
map_height_max = 1.0

# Cached map embedding
cached_map_embedding = None
cached_map_embedding_device = None

# Cost field for mass point pathfinding
terrain_cost_map = None
mass_cost_fields = {}

run_name = f"feature_based_agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(f"runs/{run_name}")
step_counter = 0

previous_healths = {}
previous_states = {}
previous_states_no_map = {}
previous_actions = {}
previous_positions = {}
previous_y_positions = {}
previous_distances = {}
previous_targets = {}
previous_action_scores = {}
previous_command_steps = {}
cancel_command_penalties = {}

lstm_hidden_states = {}
previous_lstm_hidden_states = {}

mass_destinations = {}
mass_destination_distances = {}

model_graph_logged = False

visited_mass_spots = set()
visited_mass_spots_norm = set()

consecutive_inactive = {}
last_mass_visit = {}

segment_buffers = {}
segment_stats = {}
