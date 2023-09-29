#!/bin/bash

# Input YAML folder
CONFIG_FOLDER="configs"

# Output log folder
LOG_FOLDER="logs"

# Create the log folder if it doesn't exist
mkdir -p "$LOG_FOLDER"

# Iterate over YAML files in the folder
for config_file in "$CONFIG_FOLDER"/*.yaml
do
    # Extract the base name of the YAML file without the extension
    config_name=$(basename "$config_file" .yaml)
    
    # Define the log file path
    log_file="$LOG_FOLDER/$config_name.log"

    # Run the Python script with the current YAML config file and save output to log file
    python bird_classifier_script.py --config "$config_file" >> "$log_file" 2>&1
done
