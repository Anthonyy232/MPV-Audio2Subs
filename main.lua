--[[
    AI Subtitle Service Manager for MPV
    
    This script manages the Python AI transcription service, providing:
    - Bidirectional IPC communication
    - Real-time OSD progress feedback
    - Proper lifecycle management
    - Smart subtitle selection state
]]

local mp = require 'mp'
local utils = require 'mp.utils'

-- Configuration
local config = {
    client_name = "ai_subtitle_service",
    python_script = utils.join_path(mp.get_script_directory(), 'main.py'),
    startup_timeout = 120, -- seconds to wait for model load
    osd_duration = 3000,   -- ms for normal OSD messages
}

-- Service state machine
local ServiceState = {
    IDLE = "idle",
    STARTING = "starting",
    LOADING = "loading",
    READY = "ready",
    TRANSCRIBING = "transcribing",
    ERROR = "error",
}

-- Service manager
local Service = {
    state = ServiceState.IDLE,
    progress = 0,
    total_chunks = 0,
    completed_chunks = 0,
    startup_timer = nil,
    python_running = false,
}

-- Logging helper
local function log(message)
    print("[ai_subs] " .. message)
end

-- Find Python executable in venv
-- Note: utils.join_path() only takes 2 args, must chain calls
local function get_python_executable()
    local script_dir = mp.get_script_directory()
    
    -- Try Windows path first: venv/Scripts/python.exe
    local venv_dir = utils.join_path(script_dir, 'venv')
    local scripts_dir = utils.join_path(venv_dir, 'Scripts')
    local win_path = utils.join_path(scripts_dir, 'python.exe')
    
    if utils.file_info(win_path) then
        log("Found Python at: " .. win_path)
        return win_path
    end
    
    -- Try Unix path: venv/bin/python
    local bin_dir = utils.join_path(venv_dir, 'bin')
    local unix_path = utils.join_path(bin_dir, 'python')
    
    if utils.file_info(unix_path) then
        log("Found Python at: " .. unix_path)
        return unix_path
    end
    
    log("ERROR: Python venv not found at " .. win_path .. " or " .. unix_path)
    return nil
end

-- Update OSD with current status
function Service:update_osd(message, duration)
    duration = duration or config.osd_duration
    mp.osd_message(message, duration / 1000)
end

-- Update state and log
function Service:set_state(new_state, message)
    if self.state ~= new_state then
        log("State: " .. self.state .. " -> " .. new_state)
        self.state = new_state
    end
    if message then
        log(message)
    end
end

-- Start the Python service
function Service:start()
    if self.python_running then
        log("Service already running")
        return
    end
    
    local socket_path = mp.get_property("input-ipc-server")
    local python = get_python_executable()
    
    if not socket_path then
        self:update_osd("Error: MPV IPC not configured", 5000)
        log("ERROR: input-ipc-server not set in mpv.conf")
        return
    end
    
    if not python then
        self:update_osd("Error: Python venv not found", 5000)
        return
    end
    
    self:set_state(ServiceState.STARTING, "Starting Python service...")
    self:update_osd("AI Subtitles: Starting...", 15000)
    
    -- Launch Python process
    local args = {
        python,
        "-m", "audio2subs",
        "--socket", socket_path,
    }
    
    log("Launching: " .. table.concat(args, " "))
    
    -- Launch Python process asynchronously
    mp.command_native_async({
        name = "subprocess",
        args = args,
        playback_only = false,
        capture_stdout = true,
        capture_stderr = true,
    }, function(success, result, error)
        if not success then
            log("Launch failed: " .. (error or "Unknown error"))
            
            -- Fallback to old-style invocation if package launch fails
            log("Trying fallback main.py...")
            local fallback_args = { python, config.python_script, "--socket", socket_path }
            
            mp.command_native_async({
                name = "subprocess",
                args = fallback_args,
                playback_only = false,
                capture_stdout = true,
                capture_stderr = true,
            }, function(f_success, f_result, f_error)
                if not f_success then
                    self:set_state(ServiceState.ERROR, "Failed to launch Python process (fallback also failed)")
                    self:update_osd("AI Subtitles Error: Launch failed", 5000)
                    self.python_running = false
                end
            end)
            return
        end
        
        log("Python process started successfully (PID: " .. (result.pid or "unknown") .. ")")
    end)
    
    self.python_running = true
    
    -- Set startup timeout
    self.startup_timer = mp.add_timeout(config.startup_timeout, function()
        if self.state == ServiceState.STARTING or self.state == ServiceState.LOADING then
            self:set_state(ServiceState.ERROR, "Startup timeout - model load took too long")
            self:update_osd("AI Service: Startup timeout", 5000)
        end
    end)
end

-- Stop the Python service
function Service:stop()
    if not self.python_running then
        return
    end
    
    log("Stopping service...")
    
    -- Send stop command via IPC
    mp.commandv("script-message", "ai-subs/stop")
    -- Also send old-style stop for compatibility
    mp.commandv("script-message-to", config.client_name, "ai-service-event", '"stop"')
    
    self.python_running = false
    self:set_state(ServiceState.IDLE)
    self:reset_progress()
    self:update_osd("AI Subtitles: Off", 2000)
    
    if self.startup_timer then
        self.startup_timer:kill()
        self.startup_timer = nil
    end
end

-- Toggle service on/off
function Service:toggle()
    if self.python_running then
        self:stop()
    else
        self:start()
    end
end

-- Reset progress tracking
function Service:reset_progress()
    self.progress = 0
    self.completed_chunks = 0
    self.total_chunks = 0
end

-- Handle messages from Python service
function Service:handle_message(message, ...)
    local args = {...}
    
    if message == "ai-subs/starting" then
        self:set_state(ServiceState.LOADING, "Model loading...")
        
    elseif message == "ai-subs/ready" then
        if self.startup_timer then
            self.startup_timer:kill()
            self.startup_timer = nil
        end
        self:set_state(ServiceState.READY, "Service ready")
        self:update_osd("AI Subtitles: Ready", 2000)
        
    elseif message == "ai-subs/started" then
        local filename = args[1] or "video"
        self:set_state(ServiceState.TRANSCRIBING, "Started: " .. filename)
        self:reset_progress()
        self:update_osd("AI Subtitles: Transcribing...", 3000)
        
    elseif message == "ai-subs/progress" then
        local percent = tonumber(args[1]) or 0
        local completed = tonumber(args[2]) or 0
        local total = tonumber(args[3]) or 0
        
        self.progress = percent
        self.completed_chunks = completed
        self.total_chunks = total
        
        -- Update OSD every 10%
        if percent % 10 == 0 and percent > 0 then
            self:update_osd(string.format("AI Subtitles: %d%%", percent), 1500)
        end
        
    elseif message == "ai-subs/complete" then
        self:set_state(ServiceState.READY, "Transcription complete")
        self:update_osd("AI Subtitles: Complete", 2000)
        self.progress = 100
        
    elseif message == "ai-subs/error" then
        local error_msg = args[1] or "Unknown error"
        self:set_state(ServiceState.ERROR, "Error: " .. error_msg)
        self:update_osd("AI Subtitles Error: " .. error_msg, 5000)
        
    elseif message == "ai-subs/stopped" then
        self.python_running = false
        self:set_state(ServiceState.IDLE, "Service stopped")
    end
end

-- Register message handlers
mp.register_script_message("ai-subs/starting", function() Service:handle_message("ai-subs/starting") end)
mp.register_script_message("ai-subs/ready", function() Service:handle_message("ai-subs/ready") end)
mp.register_script_message("ai-subs/started", function(f) Service:handle_message("ai-subs/started", f) end)
mp.register_script_message("ai-subs/progress", function(p, c, t) Service:handle_message("ai-subs/progress", p, c, t) end)
mp.register_script_message("ai-subs/complete", function() Service:handle_message("ai-subs/complete") end)
mp.register_script_message("ai-subs/error", function(e) Service:handle_message("ai-subs/error", e) end)
mp.register_script_message("ai-subs/stopped", function() Service:handle_message("ai-subs/stopped") end)

-- Register toggle command
mp.register_script_message("toggle_ai_subtitles", function()
    Service:toggle()
end)

-- Clean shutdown when MPV exits
mp.register_event("shutdown", function()
    Service:stop()
end)

-- Status query message
mp.register_script_message("ai-subs/status", function()
    log(string.format("Status: state=%s, progress=%d%%, running=%s",
        Service.state, Service.progress, tostring(Service.python_running)))
end)

log("AI Subtitle Service Manager loaded. Use 'toggle_ai_subtitles' to activate.")