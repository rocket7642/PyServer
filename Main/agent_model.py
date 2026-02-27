import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import map_utils
import runtime_state as state


class RTSAgent(nn.Module):
    def __init__(self, input_size, num_features):
        """Initialize the RTS agent with encoder networks, CNN map processor, LSTM, and output layers."""
        super().__init__()
        self.self_encoder = nn.Sequential(
            nn.Linear(config.SELF_FEATURES_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, config.SELF_EMBED_SIZE)
        )
        self.mass_encoder = nn.Sequential(
            nn.Linear(config.MASS_FEATURES_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, config.MASS_EMBED_SIZE)
        )
        self.map_encoder = nn.Sequential(
            nn.Linear(config.MAP_FEATURES_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, config.MAP_EMBED_SIZE)
        )
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.map_fc = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, config.MAP_EMBED_SIZE)
        )

        self._initialize_cnn_weights()

        self.friendly_unit_encoder = nn.Sequential(
            nn.Linear(config.UNIT_FEATURES_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, config.FRIENDLY_EMBED_SIZE)
        )
        self.enemy_unit_encoder = nn.Sequential(
            nn.Linear(config.ENEMY_FEATURES_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, config.ENEMY_EMBED_SIZE)
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True
        )
        self.fc1 = nn.Linear(config.LSTM_HIDDEN_SIZE, 64)
        self.fc2 = nn.Linear(64, num_features)

    def _initialize_cnn_weights(self):
        """Apply Kaiming normal init to Conv2d layers and Xavier uniform init to the map FC layer."""
        for module in self.map_cnn.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        for module in self.map_fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, hidden=None):
        """Forward pass: run encoded state through the LSTM and FC layers to produce action feature weights."""
        lstm_out, new_hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(last_out))
        feature_weights = self.fc2(x)
        if config.ENFORCE_DISTANCE_REDUCTION_NONNEG:
            idx = config.DISTANCE_REDUCTION_INDEX
            if 0 <= idx < feature_weights.shape[-1]:
                before = feature_weights[..., :idx]
                constrained = F.softplus(feature_weights[..., idx:idx+1])
                after = feature_weights[..., idx+1:]
                feature_weights = torch.cat([before, constrained, after], dim=-1)
        return feature_weights, new_hidden

    def encode_state_parts(self, agent_unit, friendly_units, enemy_units):
        """Encode self, mass, map, friendly, and enemy observations into separate embedding vectors."""
        device = next(self.parameters()).device

        unit_nx = map_utils.normalize_x(agent_unit['x'])
        unit_nz = map_utils.normalize_z(agent_unit['z'])
        unit_ny = map_utils.normalize_y(agent_unit['y'])

        # Compute nearest-enemy features for direct signal in self_encoder
        nearest_enemy_info = map_utils.find_nearest_enemy(unit_nx, unit_nz, enemy_units)
        if nearest_enemy_info is not None:
            nearest_enemy_dist, nearest_ex, nearest_ez = nearest_enemy_info
            nearest_enemy_dx = nearest_ex - unit_nx
            nearest_enemy_dz = nearest_ez - unit_nz
        else:
            nearest_enemy_dx = 0.0
            nearest_enemy_dz = 0.0
            nearest_enemy_dist = config.ENEMY_PROXIMITY_THRESHOLD

        self_features = torch.tensor([
            unit_nx,
            unit_nz,
            unit_ny,
            agent_unit['health'],
            float(len(friendly_units)),
            float(len(enemy_units)),
            nearest_enemy_dx,
            nearest_enemy_dz,
            nearest_enemy_dist,
        ], dtype=torch.float32, device=device)
        self_emb = self.self_encoder(self_features)

        if state.map_spots_norm:
            nearest_mass = min(
                state.map_spots_norm,
                key=lambda p: (p[0] - unit_nx) ** 2 + (p[1] - unit_nz) ** 2
            )
            dx = nearest_mass[0] - unit_nx
            dz = nearest_mass[1] - unit_nz
            dist = (dx ** 2 + dz ** 2) ** 0.5
        else:
            dx, dz, dist = 0.0, 0.0, 0.0
        mass_features = torch.tensor([dx, dz, dist], dtype=torch.float32, device=device)
        mass_emb = self.mass_encoder(mass_features)

        map_emb = map_utils.get_cached_map_embedding(self, device)

        friendly_vectors = []
        for u in friendly_units:
            if u['id'] == agent_unit['id']:
                continue
            friendly_vectors.append([
                map_utils.normalize_x(u['x']) - unit_nx,
                map_utils.normalize_z(u['z']) - unit_nz,
                u['health']
            ])
        if friendly_vectors:
            friendly_tensor = torch.tensor(friendly_vectors, dtype=torch.float32, device=device)
            friendly_embeds = self.friendly_unit_encoder(friendly_tensor)
            friendly_emb = torch.mean(friendly_embeds, dim=0)
        else:
            friendly_emb = torch.zeros(config.FRIENDLY_EMBED_SIZE, dtype=torch.float32, device=device)

        enemy_vectors = []
        for u in enemy_units:
            enemy_vectors.append([
                map_utils.normalize_x(u['x']) - unit_nx,
                map_utils.normalize_z(u['z']) - unit_nz,
                u['health'],
                map_utils.normalize_range(u['range']) if 'range' in u else 0.0
            ])
        if enemy_vectors:
            enemy_tensor = torch.tensor(enemy_vectors, dtype=torch.float32, device=device)
            enemy_embeds = self.enemy_unit_encoder(enemy_tensor)
            enemy_emb = torch.mean(enemy_embeds, dim=0)
        else:
            enemy_emb = torch.zeros(config.ENEMY_EMBED_SIZE, dtype=torch.float32, device=device)

        return self_emb, mass_emb, map_emb, friendly_emb, enemy_emb

    def encode_state(self, agent_unit, friendly_units, enemy_units):
        """Produce a full state vector by concatenating all encoder outputs including the map embedding."""
        self_emb, mass_emb, map_emb, friendly_emb, enemy_emb = self.encode_state_parts(
            agent_unit, friendly_units, enemy_units
        )
        state_vec = torch.cat([self_emb, mass_emb, map_emb, friendly_emb, enemy_emb], dim=0)
        state_vec = torch.nan_to_num(state_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return state_vec

    def encode_state_no_map(self, agent_unit, friendly_units, enemy_units):
        """Produce a state vector without the map embedding, for lightweight replay buffer storage."""
        self_emb, mass_emb, _, friendly_emb, enemy_emb = self.encode_state_parts(
            agent_unit, friendly_units, enemy_units
        )
        state_vec = torch.cat([self_emb, mass_emb, friendly_emb, enemy_emb], dim=0)
        state_vec = torch.nan_to_num(state_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return state_vec
