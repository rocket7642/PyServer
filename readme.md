# Capstone Archive
This archive contains the final form of the prototype for the Adaptive AI Opponent capstone project. The version within in is a prototype that displays navigation, radar build and avoidance behaviour.

The agent can train autonomously on a single machine via the included powershell script.

There are many routes that the agent can continue to be improve upon and all the files uploaded contain all components needed to run the agent.

# Requirements
The agent requires a BAR sided component so enable enable debugmode. To do so make an empty `devmode.txt` in the BAR game install directory (`/data/`), and then the `Settings/Developer` tab will appear in the lobby. Within you can disable `Simplified AI Selection` to see the added agent.

To add the agent and other contained AI, they should be placed within the current engine version's Skirmish AI Folder (`/data/engine/engine version/AI/Skirmish`). Unfortunately, on engine update you will need to shift the AIs over to the new version.

To install the widgets, place them with the BAR widget folder (`/data/luaUI/Widgets`) and enabled ingame via F11.

To install the gadgets, fork the current BAR github version into a sub-directory of the current engine version (`/data/engine/engine version/games/BAR.sdd`) and then place them within the gadgets folder (`/luarules/gadgets`) contained within the fork.

To add the maps place the compiled versions within the BAR maps folder (`/data/maps`). Any map can have its images viewed by opening like a zip file.

# Notes

The final changes to this repository focuses on fixes towards the offline training pipeline, as during the final conclusion of the project it was somewhat left behind. It should now function with respect to the current agent and be more secure for moving forward but might need improvements.