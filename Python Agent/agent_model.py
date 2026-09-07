import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import map_utils
import runtime_state as state
import unit_defs


class RTSAgent(nn.Module):
    # RTSAgent is a neural network model for real-time strategy game agents. It encodes self, economy, mass, map, and vision features, processes them through CNNs and LSTMs, and outputs action decisions.
    # The network takes as input a concatenation of encoded features and outputs action logits and move/build predictions.
    # The input features include self state, economy, mass, map, and vision embeddings.
    # The output includes action logits, move predictions, and build predictions.

    # Initialize the RTS agent with the specified input size for the LSTM.
    def __init__(self, input_size):
        super().__init__()
        self.self_encoder = nn.Sequential(
            nn.Linear(config.SELF_FEATURES_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, config.SELF_EMBED_SIZE)
        )
        self.eco_encoder = nn.Sequential(
            nn.Linear(config.ECO_FEATURES_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, config.ECO_EMBED_SIZE)
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
        self.vision_cnn = nn.Sequential(
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
        self.vision_fc = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, config.VISION_EMBED_SIZE)
        )

        # Initialize CNN weights for map and vision encoders.
        self.initialize_cnn_weights()

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
        # Set for Move only output
        # self.fc2 = nn.Linear(64, num_features)

        # New set for building/moving split
        self.action_head = nn.Linear(64, config.NUM_DISCRETE_ACTIONS)
        nn.init.uniform_(self.action_head.weight, -0.001, 0.001)
        nn.init.zeros_(self.action_head.bias)
        self.move_head = nn.Linear(64, config.NUM_ACTION_FEATURES)
        self.build_head = nn.Linear(64, config.NUM_BUILD_FEATURES)

    # Initialize CNN weights for map and vision encoders.
    # This method applies Kaiming normal initialization to all Conv2d layers
    # and Xavier uniform initialization to all Linear layers in the map and vision encoders.
    def initialize_cnn_weights(self):
        for module in self.map_cnn.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        for module in self.vision_cnn.modules():
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

        for module in self.vision_fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    # Apply softplus constraints to selected output indices to enforce known sign constraints.
    @staticmethod
    def _apply_sign_constraints(w, indices, positive=True):
        for idx in indices:
            if 0 <= idx < w.shape[-1]:
                seg = F.softplus(w[..., idx:idx+1])
                if not positive:
                    seg = -seg
                w = torch.cat([w[..., :idx], seg, w[..., idx+1:]], dim=-1)
        return w

    # Forward pass through the LSTM and fully connected layers to produce action logits and feature weights.
    def forward(self, x, hidden=None):
        lstm_out, new_hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(last_out))
        # feature_weights = self.fc2(x)
        action_logits = self.action_head(x)
        move_features = self.move_head(x)
        build_features = self.build_head(x)
        # A3: enforce known feature-weight signs via softplus (differentiable).
        move_features = self._apply_sign_constraints(move_features, config.MOVE_NONNEG_INDICES, positive=True)
        move_features = self._apply_sign_constraints(move_features, config.MOVE_NONPOS_INDICES, positive=False)
        build_features = self._apply_sign_constraints(build_features, config.BUILD_NONNEG_INDICES, positive=True)
        build_features = self._apply_sign_constraints(build_features, config.BUILD_NONPOS_INDICES, positive=False)
        return action_logits, move_features, build_features, new_hidden

    # Encode different parts of the game state into separate embedding vectors.
    def encode_state_parts(self, agent_unit, friendly_units, enemy_units):
        # Encode self, mass, map, friendly, and enemy observations into separate embedding vectors.
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
            agent_unit.get('health', 0.0),
            float(agent_unit.get('active_build_progress', 0.0)),
            float(agent_unit.get('is_constructing', 0.0)),
            float(len(friendly_units)),
            float(len(enemy_units)),
            nearest_enemy_dx,
            nearest_enemy_dz,
            nearest_enemy_dist,
        ], dtype=torch.float32, device=device)
        self_emb = self.self_encoder(self_features)

        eco_features = torch.tensor([
            float(getattr(state, 'fEnergy', 0.0)),
            float(getattr(state, 'fMass', 0.0))
        ], dtype=torch.float32, device=device)
        eco_emb = self.eco_encoder(eco_features)

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
        vision_emb = map_utils.get_vision_embedding(self, state.vision_image, device) # Error for now, in current adding phase

        friendly_vectors = []
        for u in friendly_units:
            if u['id'] == agent_unit['id']:
                continue
            friendly_vectors.append([
                map_utils.normalize_x(u['x']) - unit_nx,
                map_utils.normalize_z(u['z']) - unit_nz,
                u['health'],
                float(u.get('is_constructing', 0)),
                float(u.get('active_build_progress', 0.0)),
                map_utils.normalize_range(u.get('radar_range', 0.0)) / config.STANDARD_MAP_WIDTH,
                map_utils.normalize_range(u.get('sight_range', 0.0)) / config.STANDARD_MAP_WIDTH,
            ])
        if friendly_vectors:
            friendly_tensor = torch.tensor(friendly_vectors, dtype=torch.float32, device=device)
            friendly_embeds = self.friendly_unit_encoder(friendly_tensor)
            friendly_emb = torch.mean(friendly_embeds, dim=0)
        else:
            friendly_emb = torch.zeros(config.FRIENDLY_EMBED_SIZE, dtype=torch.float32, device=device)

        enemy_vectors = []
        for u in enemy_units:
            # Base spatial features
            evec = [
                map_utils.normalize_x(u['x']) - unit_nx,
                map_utils.normalize_z(u['z']) - unit_nz,
                u['health'],
                map_utils.normalize_range(u['range']) if 'range' in u else 0.0,
            ]
            # Weapon type one-hot (4 values)
            if 'weapon_one_hot' in u:
                evec.extend(u['weapon_one_hot'])
            else:
                # Fallback: get from unit_defs by name, or default projectile
                winfo = unit_defs.get_weapon_info(u.get('name', ''))
                evec.extend(winfo['weapon_one_hot'])
            # Normalized continuous weapon properties
            evec.append(unit_defs.normalize_projectile_speed(u.get('projectile_speed', 200)))
            evec.append(unit_defs.normalize_aoe(u.get('aoe_radius', 16)))
            evec.append(unit_defs.normalize_dps(u.get('dps', 50)))
            enemy_vectors.append(evec)
        if enemy_vectors:
            enemy_tensor = torch.tensor(enemy_vectors, dtype=torch.float32, device=device)
            enemy_embeds = self.enemy_unit_encoder(enemy_tensor)
            enemy_emb = torch.mean(enemy_embeds, dim=0)
        else:
            enemy_emb = torch.zeros(config.ENEMY_EMBED_SIZE, dtype=torch.float32, device=device)

        return self_emb, eco_emb, mass_emb, map_emb, friendly_emb, enemy_emb, vision_emb

    # Encode the full game state into a single vector by concatenating all embeddings.
    def encode_state(self, agent_unit, friendly_units, enemy_units):
        # Produce a full state vector by concatenating all encoder outputs including the map embedding.
        self_emb, eco_emb, mass_emb, map_emb, friendly_emb, enemy_emb, vision_emb = self.encode_state_parts(
            agent_unit, friendly_units, enemy_units
        )
        state_vec = torch.cat([self_emb, eco_emb, mass_emb, map_emb, friendly_emb, enemy_emb, vision_emb], dim=0)
        state_vec = torch.nan_to_num(state_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return state_vec

    # Encode the full game state into a single vector without the map embedding. 
    # This is the one used primarlly currently due to the memory cost that a long training session could have with the above method.
    def encode_state_no_map(self, agent_unit, friendly_units, enemy_units):
        # Produce a state vector without the map embedding, for lightweight replay buffer storage.
        self_emb, eco_emb, mass_emb, _, friendly_emb, enemy_emb, vision_emb = self.encode_state_parts(
            agent_unit, friendly_units, enemy_units
        )
        state_vec = torch.cat([self_emb, eco_emb, mass_emb, friendly_emb, enemy_emb, vision_emb], dim=0)
        state_vec = torch.nan_to_num(state_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return state_vec
