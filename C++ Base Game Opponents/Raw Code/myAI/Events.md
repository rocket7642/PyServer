# AI Events Reference ✅

Source: `rts/ExternalAI/Interface/AISEvents.h`

This document lists all possible AI event topics (EVENT_*) defined by the engine, the file where they originate, a short description of what they signal, and the event struct they provide (key fields).

---

## Summary

- **Origin file:** `rts/ExternalAI/Interface/AISEvents.h` 🔧
- **Enum:** `EventTopic` — all event IDs are declared here (do not change values; kept for ABI compatibility).

---

## Events (by topic)

1. **EVENT_NULL (0)**
   - Struct: none
   - Description: Placeholder/no-op event.

2. **EVENT_INIT (1)**
   - Struct: `SInitEvent`
   - Key fields: `int skirmishAIId; const SSkirmishAICallback* callback; bool savedGame;`
   - Description: Sent once as the very first event to initialize the AI instance.

3. **EVENT_RELEASE (2)**
   - Struct: `SReleaseEvent`
   - Key fields: `int reason;` (0..7 with documented meanings)
   - Description: Sent once when the AI instance is no longer needed (cleanup/finalization).

4. **EVENT_UPDATE (3)**
   - Struct: `SUpdateEvent`
   - Key fields: `int frame;`
   - Description: Sent once per game frame (main heartbeat/update tick).

5. **EVENT_MESSAGE (4)**
   - Struct: `SMessageEvent`
   - Key fields: `int player; const char* message;`
   - Description: Chat message notification from a participant (player or AI).

6. **EVENT_UNIT_CREATED (5)**
   - Struct: `SUnitCreatedEvent`
   - Key fields: `int unit; int builder;`
   - Description: A unit of this team was created (nano-frame stage).

7. **EVENT_UNIT_FINISHED (6)**
   - Struct: `SUnitFinishedEvent`
   - Key fields: `int unit;`
   - Description: A unit was fully built and is ready.

8. **EVENT_UNIT_IDLE (7)**
   - Struct: `SUnitIdleEvent`
   - Key fields: `int unit;`
   - Description: Unit has finished processing commands and has an empty queue.

9. **EVENT_UNIT_MOVE_FAILED (8)**
   - Struct: `SUnitMoveFailedEvent`
   - Key fields: `int unit;`
   - Description: Unit could not fulfill a move command (path blocked/invalid terrain/not mobile).

10. **EVENT_UNIT_DAMAGED (9)**
    - Struct: `SUnitDamagedEvent`
    - Key fields: `int unit; int attacker; float damage; float* dir_posF3; int weaponDefId; bool paralyzer;`
    - Description: A unit on this team took damage; includes attacker (may be -1) and damage direction.

11. **EVENT_UNIT_DESTROYED (10)**
    - Struct: `SUnitDestroyedEvent`
    - Key fields: `int unit; int attacker; int weaponDefID;`
    - Description: A unit of this team was destroyed.

12. **EVENT_UNIT_GIVEN (11)**
    - Struct: `SUnitGivenEvent`
    - Key fields: `int unitId; int oldTeamId; int newTeamId;`
    - Description: Unit changed ownership (give/take).

13. **EVENT_UNIT_CAPTURED (12)**
    - Struct: `SUnitCapturedEvent`
    - Key fields: `int unitId; int oldTeamId; int newTeamId;`
    - Description: Unit changed team due to capture.

14. **EVENT_ENEMY_ENTER_LOS (13)**
    - Struct: `SEnemyEnterLOSEvent`
    - Key fields: `int enemy;`
    - Description: An enemy unit entered this team's line-of-sight (LOS).

15. **EVENT_ENEMY_LEAVE_LOS (14)**
    - Struct: `SEnemyLeaveLOSEvent`
    - Key fields: `int enemy;`
    - Description: An enemy unit left this team's LOS.

16. **EVENT_ENEMY_ENTER_RADAR (15)**
    - Struct: `SEnemyEnterRadarEvent`
    - Key fields: `int enemy;`
    - Description: An enemy unit entered radar coverage of this team.

17. **EVENT_ENEMY_LEAVE_RADAR (16)**
    - Struct: `SEnemyLeaveRadarEvent`
    - Key fields: `int enemy;`
    - Description: An enemy unit left radar coverage of this team.

18. **EVENT_ENEMY_DAMAGED (17)**
    - Struct: `SEnemyDamagedEvent`
    - Key fields: `int enemy; int attacker; float damage; float* dir_posF3; int weaponDefId; bool paralyzer;`
    - Description: An enemy unit was damaged (attacker may be -1 if not visible).

19. **EVENT_ENEMY_DESTROYED (18)**
    - Struct: `SEnemyDestroyedEvent`
    - Key fields: `int enemy; int attacker;`
    - Description: An enemy unit was destroyed.

20. **EVENT_WEAPON_FIRED (19)**
    - Struct: `SWeaponFiredEvent`
    - Key fields: `int unitId; int weaponDefId;`
    - Description: Certain weapons (manual-fire/nukes/etc.) were fired by a unit.

21. **EVENT_PLAYER_COMMAND (20)**
    - Struct: `SPlayerCommandEvent`
    - Key fields: `int* unitIds; int unitIds_size; int commandTopicId; int playerId;`
    - Description: A user gave a command to units that belong to the AI's team.

22. **EVENT_SEISMIC_PING (21)**
    - Struct: `SSeismicPingEvent`
    - Key fields: `float* pos_posF3; float strength;`
    - Description: Seismic event detected (unit movement detected via seismic sensors).

23. **EVENT_COMMAND_FINISHED (22)**
    - Struct: `SCommandFinishedEvent`
    - Key fields: `int unitId; int commandId; int commandTopicId;`
    - Description: A unit finished processing a command; identifies async commands with `commandId`.

24. **EVENT_LOAD (23)**
    - Struct: `SLoadEvent`
    - Key fields: `const char* file;`
    - Description: AI should load its full state from a file.

25. **EVENT_SAVE (24)**
    - Struct: `SSaveEvent`
    - Key fields: `const char* file;`
    - Description: AI should save its full state to a file.

26. **EVENT_ENEMY_CREATED (25)**
    - Struct: `SEnemyCreatedEvent`
    - Key fields: `int enemy;`
    - Description: An enemy unit was created (nano-frame stage).

27. **EVENT_ENEMY_FINISHED (26)**
    - Struct: `SEnemyFinishedEvent`
    - Key fields: `int enemy;`
    - Description: An enemy unit finished building and is ready.

28. **EVENT_LUA_MESSAGE (27)**
    - Struct: `SLuaMessageEvent`
    - Key fields: `const char* inData;`
    - Description: Notification about a message sent by a Lua widget or unsynced gadget.

---

## Notes & Where to look in the codebase 🔎

- The authoritative declarations and comments are in `rts/ExternalAI/Interface/AISEvents.h`.
- AI implementations typically receive these events in their `handleEvent` switch/case (e.g., `AI/Skirmish/*/*AI.cpp` and `AI/Wrappers/*` wrappers). Example handled cases found in:
  - `AI/Skirmish/myAI/src/CppTestAI.cpp`
  - `AI/Skirmish/BARb/src/circuit/CircuitAI.cpp`
  - `AI/Wrappers/LegacyCpp/AIAI.cpp`
- When adding or interpreting events, preserve numeric values of `EventTopic` for ABI compatibility.

---

If you'd like, I can:
- Add this file into a more central documentation folder, or
- Generate a compact header-style reference (e.g., `Events.h` with summaries) for quick inclusion in AI code.

🔧 Let me know which option you'd prefer. 