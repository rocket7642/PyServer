--
-- Custom Options Definition Table format
--
-- A detailed example of how this format works can be found
-- in the spring source under:
-- AI/Skirmish/NullAI/data/AIOptions.lua
--
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

local options = {
	{
		key   = 'spawn_roster',
		name  = 'Spawn roster',
		desc  = 'Free-form unit roster for spawned units. Use unit names separated by +, comma, semicolon, or whitespace.',
		type  = 'string',
		def   = '',
	},
}

return options

