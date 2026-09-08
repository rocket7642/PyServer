# Capstone Archive
This archive contains the final form of the prototype for the Adaptive AI Opponent capstone project. The version within in is a prototype that displays navigation, radar build and avoidance behaviour.

There are many routes that the agent can continue to be improve upon and all the files uploaded contain all components needed to run the agent.
# Requirements
The agent requires a BAR sided component so enable enable debugmode. To do so make an empty `devmode.txt` in the BAR game install directory (`/data/`), and then the `Settings/Developer` tab will appear in the lobby. Within you can disable Simplified AI Selection to see the added agent.

To add the agent and other contained AI, they should be placed within the current engine version's Skirmish AI Folder (`/data/engine/engine version/AI/Skirmish`). Unfortunately, on engine update you will need to shift the AIs over to the new version as they are not maintained.

To install the widgets they simply are placed with the BAR widget folder (`/data/luaUI/Widgets`) and enabled ingame via F11.

To install the gadget unfortunately it is not as simple, requiring forking the current BAR github version into a sub-directory of the current engine version (`/data/engine/engine version/games/BAR.sdd`) and then placing it within the gadgets folder (`/luarules/gadgets`) contained within the fork.

To add the maps to the BAR instance place the compiled versions within the BAR maps folder (`/data/maps`). Any map can have its images viewed by opening them as if they are a zip file.