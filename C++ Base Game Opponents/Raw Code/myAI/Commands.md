# Synced Commands Reference ✅

Source: `rts/ExternalAI/Interface/AISCommands.h`

This file collects all *synced* action commands (COMMAND_*) the AI can send to the engine. Each entry lists the command topic, the struct that carries the command arguments, important fields, and a short description of what the command does.

---

## Notes
- These commands change game state (synced). Use them to move/attack/build/send messages/draw, etc.
- The authoritative source is `rts/ExternalAI/Interface/AISCommands.h` — **do not change numeric values** in `enum CommandTopic` (ABI compatibility).

---

## Quick list of commonly-used commands
- **COMMAND_SEND_TEXT_MESSAGE (6)** — `SSendTextMessageCommand` — send chat texts.
- **COMMAND_SEND_UNITS (9)** — `SSendUnitsCommand` — transfer units to another team.
- **COMMAND_DRAWER_POINT_ADD / LINE_ADD (1 / 2)** — `SAddPointDrawCommand`, `SAddLineDrawCommand` — add drawing points/lines.
- **COMMAND_DRAWER_PATH_START / PATH_DRAW_LINE (24 / 26)** — path drawer commands for drawing navigation paths.
- **COMMAND_UNIT_MOVE (42)** — `SMoveUnitCommand` — move unit to a position.
- **COMMAND_UNIT_ATTACK (45)** — `SAttackUnitCommand` — order unit to attack a unit.
- **COMMAND_UNIT_FIGHT (44)** — `SFightUnitCommand` — fight (attack-move) to a position.
- **COMMAND_UNIT_PATROL (43)** — `SPatrolUnitCommand` — patrol between waypoints.
- **COMMAND_UNIT_GUARD (47)** — `SGuardUnitCommand` — guard another unit.
- **COMMAND_UNIT_REPAIR (51)** — `SRepairUnitCommand` — repair a unit.
- **COMMAND_UNIT_LOAD_UNITS / UNLOAD (57 / 60)** — loading/unloading units.
- **COMMAND_UNIT_D_GUN / D_GUN_POS (67 / 68)** — D-gun (special weapon) commands.
- **COMMAND_UNIT_SET_MOVE_STATE (53)** — `SSetMoveStateUnitCommand` — set move state (hold fire / maneuver etc.)
- **COMMAND_UNIT_SELF_DESTROY (55)** — `SSelfDestroyUnitCommand` — self destruct.

---

## Complete reference (topic → struct → key fields)
> For full comments and reference, see `rts/ExternalAI/Interface/AISCommands.h`.

- COMMAND_NULL (0)
  - No struct (placeholder)

- COMMAND_DRAWER_POINT_ADD (1)
  - Struct: `SAddPointDrawCommand`
  - Fields: `float* pos_posF3; int lifeTime; int groupId; int ret_drawId;` — Add a point draw at position

- COMMAND_DRAWER_LINE_ADD (2)
  - Struct: `SAddLineDrawCommand`
  - Fields: `float* pos_posF3; float* pos2_posF3; int lifeTime; int groupId; int ret_drawId;` — Add a line draw

- COMMAND_SEND_START_POS (4)
  - Struct: `SSendStartPosCommand`
  - Fields: `float* pos_posF3;` — Send AI start position

- COMMAND_SEND_TEXT_MESSAGE (6)
  - Struct: `SSendTextMessageCommand`
  - Fields: `const char* text; int zone;` — Chat message

- COMMAND_SET_LAST_POS_MESSAGE (7)
  - Struct: `SSetLastPosMessageCommand`
  - Fields: `float* pos_posF3;` — Assign a map location to last text

- COMMAND_SEND_RESOURCES (8)
  - Struct: `SSendResourcesCommand`
  - Fields: `int resourceId; float amount; int receivingTeamId; bool ret_isExecuted;` — Transfer resources

- COMMAND_SEND_UNITS (9)
  - Struct: `SSendUnitsCommand`
  - Fields: `int* unitIds; int unitIds_size; int receivingTeamId; int ret_sentUnits;` — Transfer units to team

- COMMAND_PATH_INIT (16)
  - Struct: `SInitPathCommand`
  - Fields: `float* start_posF3; float* end_posF3; int pathType; float goalRadius; int ret_pathId;` — Path-finder: init

- COMMAND_PATH_GET_APPROXIMATE_LENGTH (17)
  - Struct: `SGetApproximateLengthPathCommand`
  - Fields: `float* start_posF3; float* end_posF3; int pathType; float goalRadius; float ret_approximatePathLength;`

- COMMAND_PATH_GET_NEXT_WAYPOINT (18)
  - Struct: `SGetNextWaypointPathCommand`
  - Fields: `int pathId; int ret_x; int ret_y; int ret_z;` — Get next waypoint of path

- COMMAND_PATH_FREE (19)
  - Struct: `SFreePathCommand`
  - Fields: `int pathId;` — Free path resources

- COMMAND_CALL_LUA_RULES (21)
  - Struct: `SCallLuaRulesCommand` — call into LuaRules

- COMMAND_DRAWER_ADD_NOTIFICATION (22)
  - Struct: `SAddNotificationDrawerCommand` — show notification

- COMMAND_DRAWER_DRAW_UNIT (23)
  - Struct: `SDrawUnitDrawerCommand` — attach draw to unit

- COMMAND_DRAWER_PATH_START / FINISH / DRAW_LINE (24 / 25 / 26)
  - Structs: `SStartPathDrawerCommand`, `SFinishPathDrawerCommand`, `SDrawLinePathDrawerCommand` — Path drawer operations

- COMMAND_UNIT_BUILD (35)
  - Struct: `SBuildUnitCommand`
  - Fields: `int builderId; int unitDefId; float* pos_posF3; int facing; int ret_newUnitId;` — Build a unit at position

- COMMAND_UNIT_STOP (36)
  - Struct: `SStopUnitCommand`
  - Fields: `int unitId;` — Stop unit

- COMMAND_UNIT_MOVE (42)
  - Struct: `SMoveUnitCommand`
  - Fields: `int unitId; float* pos_posF3; int options; int timeOut;` — Move unit to position

- COMMAND_UNIT_PATROL (43)
  - Struct: `SPatrolUnitCommand`
  - Fields: `int unitId; float* pos_posF3;` — Patrol to position(s)

- COMMAND_UNIT_FIGHT (44)
  - Struct: `SFightUnitCommand`
  - Fields: `int unitId; float* pos_posF3;` — Fight (attack-move) to position

- COMMAND_UNIT_ATTACK (45)
  - Struct: `SAttackUnitCommand`
  - Fields: `int unitId; int targetUnitId;` — Attack target unit

- COMMAND_UNIT_ATTACK_AREA (46)
  - Struct: `SAttackAreaUnitCommand`
  - Fields: `int unitId; float* pos_posF3; float radius;` — Attack an area

- COMMAND_UNIT_GUARD (47)
  - Struct: `SGuardUnitCommand`
  - Fields: `int unitId; int guardedUnitId;` — Guard other unit

- COMMAND_UNIT_REPAIR (51)
  - Struct: `SRepairUnitCommand`
  - Fields: `int unitId; int repairedUnitId;` — Repair unit

- COMMAND_UNIT_SET_FIRE_STATE (52)
  - Struct: `SSetFireStateUnitCommand`
  - Fields: `int unitId; int fireState;` — Change fire state

- COMMAND_UNIT_SET_MOVE_STATE (53)
  - Struct: `SSetMoveStateUnitCommand`
  - Fields: `int unitId; int moveState;` — Change move behavior

- COMMAND_UNIT_SELF_DESTROY (55)
  - Struct: `SSelfDestroyUnitCommand`
  - Fields: `int unitId;` — Self-destruct

- COMMAND_UNIT_LOAD_UNITS / UNLOAD_UNITS (57 / 60)
  - Structs: `SLoadUnitsUnitCommand`, `SUnloadUnitsAreaUnitCommand`, etc. — load/unload units

- COMMAND_UNIT_D_GUN / D_GUN_POS (67 / 68)
  - Struct: `SDGunUnitCommand`, `SDGunPosUnitCommand`
  - Fields: `int unitId; int targetId; float* pos_posF3;` — D-gun fire

- COMMAND_UNIT_CUSTOM (78)
  - Struct: `SCustomUnitCommand` — Some unit custom command; includes parameters

- COMMAND_PAUSE (81)
  - Struct: `SPauseCommand` — Pause the game

- COMMAND_TRACE_RAY / FEATURE_TRACE_RAY (80 / 95)
  - Structs: `STraceRayCommand`, `SFeatureTraceRayCommand` — Query ray traces

- COMMAND_CALL_LUA_UI (96)
  - Struct: `SCallLuaUICommand` — Call Lua UI function

(There are many more specialized commands for debug drawing, overlay textures, group management, pathing utilities and cheats — see full file.)

---

## How to use these from your AI
- Use `SSkirmishAICallback.handleCommand(commandTopicId, commandData)` (or wrapper functions like `GiveOrder`, `GiveGroupOrder`) to send commands to the engine.
- For unit commands, wrappers such as `GiveOrder(int unitId, Command* c)` accept a `Command` object — create the appropriate `Command` with topic and parameters.
- Check return fields in command structs (e.g., `ret_newUnitId`, `ret_pathId`) for results.
- Respect `UnitCommandOptions` bitflags to apply SHIFT/CTRL/ALT behaviour when queueing commands.

---

## Want a generated cheat-sheet or header?
I can:
- Generate a compact `Commands.md` with every command expanded (full field lists), or
- Create a small C++ header `CommandsRef.h` with brief typedefs and comments for easy inclusion in your AI code.

Which would you prefer? 🔧