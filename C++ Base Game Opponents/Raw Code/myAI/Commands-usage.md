# Commands — Usage notes & differences

This short note explains the practical difference between in-process unit methods (e.g., `Unit->MoveTo` / `unit->CmdMoveTo`) and the external Skirmish AI command `COMMAND_UNIT_MOVE`.

- **Unit->MoveTo / unit->CmdMoveTo**
  - These are convenience methods in in-process AI code and wrappers.
  - They typically call the engine's internal command execution API (e.g., `ExecuteCustomCommand` with `CMD_RAW_MOVE` or similar `CMD_*` ids) or directly create engine `Command` objects.
  - Use these when writing C++ AI logic that runs inside the engine or uses the provided wrapper classes (they may expose engine-only features and convenience behavior).

- **COMMAND_UNIT_MOVE / `SMoveUnitCommand`**
  - This is the official external/ABI-level command (topic id 42) used by Skirmish AIs via `handleCommand()` / `GiveOrder()`.
  - External AIs must populate an `SMoveUnitCommand` and send it; the engine translates it into internal commands to move the unit.
  - Use this for out-of-process/remote AIs or whenever ABI stability and explicit command structs are required.

- **Summary**
  - Both mechanisms cause a unit to move, but they live in different API layers: internal wrappers vs external Skirmish AI interface. The internal wrappers are more convenient inside engine code; `COMMAND_UNIT_MOVE` is the stable, documented external interface.

---

Files:
- `Commands.md` (compact summary)
- `Commands-full.md` (full exhaustive structs — verbatim excerpt)
- `Commands-usage.md` (this file — short usage notes)

If you want, I can:
- Update `Commands.md` to point directly to `Commands-full.md` and `Commands-usage.md`, or
- Generate a `CommandsRef.h` header with small helpers for sending common commands from your AI code.

Which would you like next? 👇