function gadget:GetInfo()
    return { name = "AI GameOver Notifier", layer = 1, enabled = true }
end

if gadgetHandler:IsSyncedCode() then return false end

function gadget:GameOver(winningAllyTeams)
    -- Send to all AI teams (-1 = broadcast to all)
    Spring.SendSkirmishAIMessage(-1, "GAMEOVER")
end