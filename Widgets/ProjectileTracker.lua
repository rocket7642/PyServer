
function widget:GetInfo()
    return {
        name = "ProjectileTracker",
        desc = "Tracks visible projectiles, destination, and path",
        author = "You",
        date = "2025",
        license = "GPL",
        layer = 0,
        enabled = true
    }
end

local projectilesData = {}  -- Table to store recorded data: {projectileID = {destination = {x,y,z}, path = {{x,y,z}, ...}}}

function widget:Update(dt)
    projectilesData = {}  -- Reset each frame for live tracking; or accumulate if you want history
    
    -- Get map bounds
    local mapX = Spring.mapSizeX
    local mapZ = Spring.mapSizeZ
    
    -- Get local player's ally team for LOS check
    local allyTeamID = Spring.GetLocalAllyTeamID()
    
    -- Get all projectiles on the map
    local projectileIDs = Spring.GetProjectilesInRectangle(0, 0, mapX, mapZ)
    if not projectileIDs then return end
    
    for _, projectileID in ipairs(projectileIDs) do
        -- Get position and check visibility
        local px, py, pz = Spring.GetProjectilePosition(projectileID)
        if px and Spring.IsPosInLos(px, py, pz, allyTeamID) then
            -- Get projectile info for destination
            local ownerID, weaponDefID, isBomb, isDgun, isShot, isStunned, isWobbling, isTerrain, targetType, targetID, startX, startY, startZ, endX, endY, endZ = Spring.GetProjectileInfo(projectileID)
            
            -- Record destination
            local destination = {x = endX, y = endY, z = endZ}
            
            -- Record/simulate path: Start with current position, then predict future positions
            local path = {{x = px, y = py, z = pz}}  -- Current position as path start
            
            -- Get velocity and gravity for trajectory simulation
            local vx, vy, vz = Spring.GetProjectileVelocity(projectileID)
            local gravity = Spring.GetProjectileGravity(projectileID) or 0  -- Default to 0 if not available
            
            -- Simulate path for next 5 seconds (adjust as needed)
            local simulationTime = 5  -- seconds
            local steps = 50  -- Number of prediction steps
            local timeStep = simulationTime / steps
            for i = 1, steps do
                local t = i * timeStep
                local futureX = px + vx * t
                local futureY = py + vy * t - 0.5 * gravity * t * t  -- Ballistic: y = y0 + vy*t - 0.5*g*t^2
                local futureZ = pz + vz * t
                table.insert(path, {x = futureX, y = futureY, z = futureZ})
            end
            
            -- Store in data table
            projectilesData[projectileID] = {
                destination = destination,
                path = path
            }
        end
    end
    
    -- Optional: Log or display data (e.g., for debugging)
    -- Spring.Echo("Visible projectiles: " .. #projectilesData)
    -- for id, data in pairs(projectilesData) do
    --     Spring.Echo("Projectile " .. id .. " dest: " .. tostring(data.destination.x) .. "," .. tostring(data.destination.y) .. "," .. tostring(data.destination.z))
    -- end
end

-- Optional: Draw paths on screen for visualization (requires OpenGL calls)
function widget:DrawWorld()
    gl.Color(1, 0, 0, 0.5)  -- Red semi-transparent
    for _, data in pairs(projectilesData) do
        -- Draw destination as a point
        gl.PushMatrix()
        gl.Translate(data.destination.x, data.destination.y, data.destination.z)
        gl.BeginEnd(GL.POINTS, function() gl.Vertex(0,0,0) end)
        gl.PopMatrix()
        
        -- Draw path as a line
        gl.BeginEnd(GL.LINE_STRIP, function()
            for _, pos in ipairs(data.path) do
                gl.Vertex(pos.x, pos.y, pos.z)
            end
        end)
    end
end