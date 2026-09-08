function widget:GetInfo()
   return {
      name         = "ActionTrackerLive",
      desc         = "Tracks all actions during a match, saves to file once match ends",
      author       = "Rocket7642",
      date         = "12/28/2025",
      license      = "GPLv2 or later",
      layer        = 0,
      enabled      = true
   }
end

local playerLists = {}
local saveLocation = "LuaUI/SavedInfo" -- A custom folder for saving information (make sure exists)
local saveDirectory
local GetPlayerRoster = Spring.GetPlayerRoster

local mapX = 0
local mapZ = 0

local massPoints = {}
local hydros = {}
local waterLevel = 0
local myTeam = Spring.GetMyTeamID()

-- Export data storage
local unitListCommands = {}
local unitListEnemy = {}
local unitListFriendly = {}
local unitListKnownEnemy = {}
local filePrefix = ""  -- Will be set during Initialize

local APIAccess 

function fakeUnit(unitID, Range, Health, Position)
    return {
        id = unitID,
        range = Range,
        health = Health,
        position = Position
    }
end

function fakeCommand(unitID, CmdID, CmdParams, CmdOptions)
    return {
        id = unitID,
        cmdID = CmdID,
        cmdParams = CmdParams,
        cmdOptions = CmdOptions
    }
end

-- This widget is purely for recording the data during the match to a file. The system of nodes is not required here.
-- Leave that for a separate python program that can process it into a form that works
-- Returns all weapon ranges of a given unitDefID
-- Pulled from rangefinder widget
local function GetWeaponRanges(unitDefID)
  local weapons = UnitDefs[unitDefID].weapons
  local ranges = {}
  local groundRanges = {}
  for idx = 1, #weapons do
    local weaponDefID = weapons[idx].weaponDef
    local weapon = WeaponDefs[weaponDefID]
    if weapon.range then 
      if weapon.canAttackGround then table.insert(groundRanges, weapon.range) end
      table.insert(ranges, weapon.range)
    end
  end

  -- ground ranges are used for sorting if config is set to "maxGroundRange",
  -- but still need to display air ranges if AA unit ghost is displayed
  return ranges, groundRanges 
  
end

function widget:UnitCommand(unitID, unitDefID, teamID, cmdID, cmdParams, cmdOptions, cmdTag)
    if(not cmdOptions.shift) then -- clear table as all other commands have been cancelled
        unitListCommands = {}
    end
    local cmd = fakeCommand(unitID, cmdID, cmdParams, cmdOptions)
    -- add command to command list
    table.insert(unitListCommands, cmd)

    Spring.Echo("UnitID: " .. unitID .. " CmdID: " .. cmdID .. " CmdParams: " .. table.concat(cmdParams, ",") .. " CmdOptions: " .. table.concat(cmdOptions, ","))
end

function widget:UnitCmdDone(unitID, unitDefID, unitTeam, cmdID, cmdParams, cmdOpts, cmdTag)
    -- remove command from command list (scan for comparable command, use cmdID)
    for i, cmd in ipairs(unitListCommands) do
        if cmd.id == unitID and cmd.cmdID == cmdID then
            table.remove(unitListCommands, i)
            break
        end
    end

    Spring.Echo("Finished UnitID: " .. unitID .. " CmdID: " .. cmdID .. " CmdParams: " .. table.concat(cmdParams, ",") .. " CmdOptions: " .. table.concat(cmdOpts, ","))

end


function widget:UnitEnteredLos(unitID, unitTeam, allyTeam, unitDefID)
    -- add to enemy list if not already present
    local alreadyPresent = false
    for i, v in ipairs(unitListEnemy) do
        if v == unitID then
            alreadyPresent = true
            break
        end
    end
    if not alreadyPresent then
        table.insert(unitListEnemy, unitID)
    end
    -- add to known enemy list if not already present
    alreadyPresent = false
    for i, v in ipairs(unitListKnownEnemy) do
        if v == unitID then
            alreadyPresent = true
            break
        end
    end
    if not alreadyPresent then
        table.insert(unitListKnownEnemy, unitID)
    end
    Spring.Echo("Unit Entered LOS: " .. unitID .. " from team " .. unitTeam)
end

function widget:UnitLeftLos(unitID, unitTeam, allyTeam, unitDefID)
    -- remove from enemy list if present
    for i, v in ipairs(unitListEnemy) do
        if v == unitID then
            table.remove(unitListEnemy, i)
            break
        end
    end
    Spring.Echo("Unit Left LOS: " .. unitID .. " from team " .. unitTeam)
end

function widget:UnitDestroyed(unitID, unitDefID, teamID)
    -- remove from known enemy list and enemy list if present
    for i, v in ipairs(unitListKnownEnemy) do
        if v == unitID then
            table.remove(unitListKnownEnemy, i)
            break
        end
    end
    for i, v in ipairs(unitListEnemy) do
        if v == unitID then
            table.remove(unitListEnemy, i)
            break
        end
    end
    Spring.Echo("Unit Destroyed: " .. unitID .. " from team " .. teamID)
end

function widget:UnitCreated(unitID, unitDefID, teamID)
    -- Save commander to friendly list
    if teamID == myTeam then
        table.insert(unitListFriendly, unitID)
    end
    Spring.Echo("Unit Created: " .. unitID .. " for team " .. teamID)
end

function widget:Initialize()
    -- Cannot be used in a replay as LOS is require for use.
    if (Spring.IsReplay()) then -- disable widget in normal games
        Spring.Echo("TestWidget disabled: only for gameplay")
        widgetHandler:RemoveWidget()
        return
    end

    -- Cannot be used in spectator mode as LOS is required for use
    local _, _, spectator = Spring.GetPlayerInfo(Spring.GetMyPlayerID())
    if spectator then
        Spring.Echo("TestWidget disabled: not for spectators")
        widgetHandler:RemoveWidget()
        return
    end

    -- -- Find correct team to watch
    -- local players = Spring.GetPlayerList(0)
    -- -- Scan the players to verify who isnt an AI
    -- for i = 1, #players do
    --     local pID = players[i]
    --     local name, active, spectator, teamID, allyTeamID = Spring.GetPlayerInfo(pID)
    --     if not spectator then -- This line will change when run of others replays
    --         myTeam = teamID
    --         break
    --     end
    -- end
    -- Spring.Echo("Watching team: " .. tostring(myTeam))

    myTeam = Spring.GetMyTeamID()

    local new_file = assert(io.open("test.txt", "w"), "Unable to save file")
    new_file:write("Check Check Check")
    new_file:close()
    Spring.Echo("TestWidget initialized")

    -- Timer for once-per-second updates (every 1 second)
    widget.updateTimer = 0
    widget.updateInterval = 1  -- 1 seconds = once per second

    -- Fetch map data (must be done in Initialize, not at module load)
    waterLevel = Spring.GetWaterLevel(500, 500)

    -- Get map bounds
    mapX = Game.mapSizeX
    mapZ = Game.mapSizeZ

    -- Verify if mapSizes were successfully retrieved
    if not mapX or not mapZ then
        Spring.Echo("Error: Could not retrieve map size")
        widgetHandler:RemoveWidget()
        mapX = 0
        mapZ = 0
    end

    APIAccess = WG.resource_spot_finder

    if not APIAccess then
        Spring.Echo("[Key Tracker] Error: resource_spot_finder not found! Removing self.")
        widgetHandler:RemoveWidget(self)
        return
    end
    
    -- Try different API methods for getting resource points
    if APIAccess then
        -- Using the resource_spot_finder gadget which finds actual resource spots
        massPoints = APIAccess.metalSpotsList
        hydros = APIAccess.geoSpotsList
        Spring.Echo("Table length is: " .. tostring(#massPoints)) -- Debug output to verify if anything is in spots
        
        Spring.Echo("Resource spots loaded from gadget")
    else
        Spring.Echo("Warning: Could not find resource point API methods")
        widgetHandler:RemoveWidget()
        massPoints = {}
        hydros = {}
    end

    -- Create filename prefix based on map name and timestamp
    local mapName = Game.mapName or "map"
    mapName = mapName:match("([^/\\]+)$"):sub(1, -5) -- Remove file extension and path
    local timestamp = os.date("%Y%m%d_%H%M%S")
    filePrefix = mapName .. "_" .. timestamp
    
    -- Export static map data once at initialization
    ExportMapHeights(filePrefix)
    ExportMapSpots(filePrefix)
    
    Spring.Echo("Exports initialized, saving to: " .. saveLocation)
end

function ExportMapHeights(filePrefix)
    -- Export map heights as CSV
    local file = io.open(saveLocation .. "/" .. filePrefix .. "_map_heights.csv", "w")
    if not file then
        Spring.Echo("Error: Could not create map_heights.csv")
        return
    end
    
    -- Write header with metadata
    file:write("MapSizeX," .. mapX .. "\n")
    file:write("MapSizeZ," .. mapZ .. "\n")
    file:write("WaterLevel," .. waterLevel .. "\n")
    file:write("x,z,height\n")

    -- Write heights
    for z = 0, mapZ/8 - 1 do
        for x = 0, mapX/8 - 1 do
            local height = Spring.GetGroundHeight(x*8, z*8)  -- Get height at (x, z)
            file:write(string.format("%d,%d,%.2f\n", x*8, z*8, height))
        end
    end
    
    file:close()
    Spring.Echo("Map heights exported to CSV")
end

function ExportMapSpots(filePrefix)
    -- Export mass points and hydro spots as CSV
    local file = io.open(saveLocation .. "/" .. filePrefix .. "_map_spots.csv", "w")
    if not file then
        Spring.Echo("Error: Could not create map_spots.csv")
        return
    end
    
    -- Write header
    file:write("type,index,x,z,worth\n")
    
    -- Write mass extraction points
    for i, spot in ipairs(massPoints) do
        file:write(string.format("metal,%d,%.2f,%.2f,%.2f\n", i, spot.x, spot.z, spot.worth))
    end
    
    -- Write hydro/geo points
    for i, spot in ipairs(hydros) do
        file:write(string.format("geo,%d,%.2f,%.2f\n", i, spot.x, spot.z))
    end
    
    file:close()
    Spring.Echo("Map spots exported to CSV")
end


function ExportUpdateData()
    -- Export data collected during updates
    local file = io.open(saveLocation .. "/" .. filePrefix .. "_runtime_data.txt", "a")
    if not file then
        Spring.Echo("Error: Could not create runtime_data.txt")
        return
    end
    
    file:write("UPDATE: " .. Spring.GetGameSeconds() .. "\n")
    
    -- Commands
    file:write("Commands:\n")
    for i, cmd in ipairs(unitListCommands) do
        file:write(string.format("Command %d: UnitID: %d, CmdID: %d, CmdParams: %s, CmdOptions: %s\n", i, cmd.id, cmd.cmdID, table.concat(cmd.cmdParams, ","), table.concat(cmd.cmdOptions, ",")))
    end
    unitListCommands = {}  -- Clear commands after export
    
    -- Enemy units
    file:write("Enemy Units:\n")
    for i, unit in ipairs(unitListEnemy) do
        local range = Spring.GetUnitMaxRange(unit) or 0
        local health = Spring.GetUnitHealth(unit)
        local x, y, z = Spring.GetUnitPosition(unit)
        file:write(string.format(
            "Enemy %d: %d, Range: %.2f, Health: %.2f, Position: (%.2f, %.2f, %.2f)\n",
            i, unit, range, health or 0, x or 0, y or 0, z or 0
        ))
    end
    
    -- Friendly units
    file:write("Friendly Units:\n")
    for i, unit in ipairs(unitListFriendly) do
        Spring.Echo("Exporting friendly unit: " .. unit)  -- Debug output
        local range = Spring.GetUnitMaxRange(unit) or 0
        local health = Spring.GetUnitHealth(unit)
        local x, y, z = Spring.GetUnitPosition(unit)
        file:write(string.format(
            "Friendly %d: %d, Range: %.2f, Health: %.2f, Position: (%.2f, %.2f, %.2f)\n",
            i, unit, range, health or 0, x or 0, y or 0, z or 0
        ))
    end
    
    -- Known enemy units
    file:write("Known Enemy Units:\n")
    for i, unit in ipairs(unitListKnownEnemy) do
        local range = Spring.GetUnitMaxRange(unit) or 0
        local health = Spring.GetUnitHealth(unit)
        local x, y, z = Spring.GetUnitPosition(unit)
        file:write(string.format(
            "Known Enemy %d: %d, Range: %.2f, Health: %.2f, Position: (%.2f, %.2f, %.2f)\n",
            i, unit, range, health or 0, x or 0, y or 0, z or 0
        ))
    end
    
    file:write("END\n")
    file:close()
end

function widget:Update(deltaTime)
    widget.updateTimer = widget.updateTimer + deltaTime
    
    if widget.updateTimer >= widget.updateInterval then
        widget.updateTimer = 0  -- Reset timer
        
        -- Populate export data arrays is handled in widget callbacks
        
        -- Export collected data
        ExportUpdateData()
        
        Spring.Echo("Information Stored")
    end
end

function widget:Shutdown()
    -- close file writes if needed
    Spring.Echo("TestWidget shutdown")
end

-- Get map data at start (height, slope, water level)
-- Update map data on large impacts
-- Store units (hp, location, range), projectiles (speed, destination), actions (unit, action, destination) in tuple
-- Link tuple together from instance to instance (every second, every half second, etc)
