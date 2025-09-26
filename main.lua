local mp = require 'mp'
local utils = require 'mp.utils'

mp.commandv("script-binding", "ai_service_manager")

-- ===================================================================
-- CONFIGURATION
-- ===================================================================
local config = {
    CLIENT_NAME = "ai_subtitle_service",
    PYTHON_SCRIPT_PATH = utils.join_path(mp.get_script_directory(), 'main.py'),
    DOCKER_IMAGE_NAME = "mpv-whisperx-server:latest",
    DOCKER_CONTAINER_NAME = "mpv_whisperx_instance"
}

-- ===================================================================
-- UTILITIES
-- ===================================================================
local function log(message)
    print("[ai_service_manager] " .. message)
end

local function get_python_executable()
    -- Logic to find the Python executable inside the virtual environment
    local script_dir = mp.get_script_directory()
    local win_path = script_dir .. '\\venv\\Scripts\\python.exe'
    if utils.file_info(win_path) then return win_path end
    local unix_path = script_dir .. '/venv/bin/python'
    if utils.file_info(unix_path) then return unix_path end
    return nil
end

-- ===================================================================
-- SERVICE MANAGER LOGIC (Persistent Docker Model)
-- ===================================================================
local Service = {}
Service.__index = Service

function Service:new()
    local instance = setmetatable({}, Service)
    instance.python_service_running = false
    instance.docker_container_running = false
    -- Ensure a clean state by stopping any leftover container on startup.
    self:stop_docker_container() 
    return instance
end

function Service:build_docker_image_if_needed()
    log("Checking if Docker image '" .. config.DOCKER_IMAGE_NAME .. "' exists...")
    local check_result = utils.subprocess({ args = { "docker", "images", "-q", config.DOCKER_IMAGE_NAME }, cancellable = false })

    if check_result.status == 0 and check_result.stdout and string.match(check_result.stdout, "%S") then
        log("Docker image already exists. Skipping build.")
        return true
    end

    log("Docker image not found. Building it now...")
    mp.osd_message("AI Subtitles: Building Docker image...", 10)

    local script_dir = mp.get_script_directory()
    local build_command
    if string.match(script_dir, "\\") then
        -- Windows command structure
        build_command = 'cd /d "' .. script_dir .. '" && docker build -t "' .. config.DOCKER_IMAGE_NAME .. '" .'
        build_result = utils.subprocess({ args = { "cmd.exe", "/c", build_command }, cancellable = false })
    else
        -- Unix command structure
        build_command = 'cd "' .. script_dir .. '" && docker build -t "' .. config.DOCKER_IMAGE_NAME .. '" .'
        build_result = utils.subprocess({ args = { "sh", "-c", build_command }, cancellable = false })
    end

    if build_result.status ~= 0 then
        local stdout_msg = build_result.stdout or "(no stdout)"
        local stderr_msg = build_result.stderr or "(no stderr)"
        log("FATAL: Docker build failed. Stdout: " .. stdout_msg .. " Stderr: " .. stderr_msg)
        mp.osd_message("Error: AI Docker image build failed!", 5)
        return false
    end

    log("Docker image built successfully.")
    mp.osd_message("AI Docker image built successfully.", 3)
    return true
end

function Service:start_docker_container()
    log("Starting persistent Docker container...")
    local run_result = utils.subprocess({
        args = {
            "docker", "run",
            "-d",
            "--name", config.DOCKER_CONTAINER_NAME,
            "--gpus", "all",
            "-p", "127.0.0.1:5000:5000",
            "-v", "whisperx_cache:/.cache",
            "-e", "WHISPER_MODEL=large-v3",
            config.DOCKER_IMAGE_NAME
        },
        cancellable = false
    })

    if run_result.status ~= 0 then
        local stderr_msg = run_result.stderr or "(no stderr)"
        log("Error starting Docker container: " .. stderr_msg)
        mp.osd_message("Error starting AI container!", 5)
        return false
    end
    
    log("Docker container started.")
    self.docker_container_running = true
    return true
end

function Service:stop_docker_container()
    log("Stopping and removing Docker container '" .. config.DOCKER_CONTAINER_NAME .. "'...")
    -- Use 'docker rm -f' to forcefully stop and remove the container in one step.
    utils.subprocess({ args = { "docker", "rm", "-f", config.DOCKER_CONTAINER_NAME }, cancellable = false })
    log("Container cleanup command sent.")
    self.docker_container_running = false
end

function Service:start_python_service()
    if self.python_service_running then return end
    log("Starting Python client service...")
    local socket_path = mp.get_property("input-ipc-server")
    local python_executable = get_python_executable()
    if not (socket_path and python_executable) then 
        log("FATAL: MPV socket path or Python executable not found.")
        return 
    end
    -- Start the Python client detached so it runs independently of the Lua script.
    utils.subprocess_detached({ args = { python_executable, config.PYTHON_SCRIPT_PATH, "--socket", socket_path } })
    self.python_service_running = true
end

function Service:stop_python_service()
    if not self.python_service_running then return end
    log("Stopping Python client service via IPC message...")
    -- Send an IPC message to the Python client to trigger a graceful shutdown.
    mp.commandv("script-message-to", config.CLIENT_NAME, "ai-service-event", '"stop"')
    self.python_service_running = false
end

function Service:toggle()
    if self.docker_container_running then
        self:stop_python_service()
        self:stop_docker_container()
        mp.osd_message("AI Subtitle Service: OFF", 2)
    else
        if not self:build_docker_image_if_needed() then return end
        if not self:start_docker_container() then return end
        -- Display a long OSD message because the model loading takes time,
        -- which happens immediately after the container starts.
        mp.osd_message("AI Service: Starting... (Loading model, please wait)", 20)
        self:start_python_service()
        mp.osd_message("AI Subtitle Service: ON", 2)
    end
end

-- ===================================================================
-- INITIALIZATION AND EVENT BINDING
-- ===================================================================
local AIService = Service:new()

mp.register_script_message("toggle_ai_subtitles", function()
    AIService:toggle()
end)

mp.register_event("shutdown", function()
    -- Ensure all external processes are cleaned up when MPV closes.
    AIService:stop_python_service()
    AIService:stop_docker_container()
end)

log("Script loaded. Use 'toggle_ai_subtitles' script-message to activate.")