local mp = require 'mp'
local utils = require 'mp.utils'

mp.commandv("script-binding", "ai_service_manager")

-- Configuration shared between Lua and Python so paths and names match
local config = {
    CLIENT_NAME = "ai_subtitle_service",
    PYTHON_SCRIPT_PATH = utils.join_path(mp.get_script_directory(), 'main.py')
}

local function log(message)
    -- Consistent log prefix to make script messages easy to find
    print("[ai_service_manager] " .. message)
end

local function get_python_executable()
    -- Prefer a virtualenv python to match the installed dependencies
    local script_dir = mp.get_script_directory()
    local win_path = script_dir .. '\\venv\\Scripts\\python.exe'
    if utils.file_info(win_path) then return win_path end
    local unix_path = script_dir .. '/venv/bin/python'
    if utils.file_info(unix_path) then return unix_path end
    log("ERROR: Could not find Python executable in venv. Please run install script.")
    return nil
end

-- Orchestrate a detached Python client process so the MPV UI remains responsive
local Service = {}
Service.__index = Service

function Service:new()
    local instance = setmetatable({}, Service)
    instance.python_service_running = false
    return instance
end

function Service:start_python_service()
    if self.python_service_running then return end
    log("Starting Python client service...")
    mp.osd_message("AI Service: Starting... (Loading model, please wait)", 15)

    local socket_path = mp.get_property("input-ipc-server")
    local python_executable = get_python_executable()

    if not (socket_path and python_executable) then 
        log("FATAL: MPV socket path or Python executable not found.")
        mp.osd_message("Error: AI service cannot start. Run install script.", 5)
        return 
    end

    utils.subprocess_detached({ args = { python_executable, config.PYTHON_SCRIPT_PATH, "--socket", socket_path } })
    self.python_service_running = true
end

function Service:stop_python_service()
    if not self.python_service_running then return end
    log("Stopping Python client service via IPC message...")
    mp.commandv("script-message-to", config.CLIENT_NAME, "ai-service-event", '"stop"')
    self.python_service_running = false
end

function Service:toggle()
    if self.python_service_running then
        self:stop_python_service()
        mp.osd_message("AI Subtitle Service: OFF", 2)
    else
        self:start_python_service()
    end
end

local AIService = Service:new()

mp.register_script_message("toggle_ai_subtitles", function()
    AIService:toggle()
end)

mp.register_event("shutdown", function()
    -- Ensure service is stopped when MPV exits to avoid orphan processes
    AIService:stop_python_service()
end)

log("Script loaded. Use 'toggle_ai_subtitles' script-message to activate.")