#!/bin/bash

# lambda.sh - Setup script for PATH export and symlink creation
# This script is idempotent and safe to run multiple times

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_info() {
    print_status "$YELLOW" "[INFO] $1"
}

print_success() {
    print_status "$GREEN" "[SUCCESS] $1"
}

print_error() {
    print_status "$RED" "[ERROR] $1"
}

# Function to add PATH export to .bashrc
setup_path_export() {
    local bashrc_file="/home/ubuntu/.bashrc"
    local path_export='export PATH="/home/ubuntu/.pixi/bin:$PATH"'
    
    print_info "Checking PATH export in $bashrc_file..."
    
    # Create .bashrc if it doesn't exist
    if [[ ! -f "$bashrc_file" ]]; then
        print_info "Creating $bashrc_file as it doesn't exist"
        touch "$bashrc_file"
    fi
    
    # Check if the exact line already exists
    if grep -Fxq "$path_export" "$bashrc_file"; then
        print_success "PATH export already exists in $bashrc_file"
    else
        print_info "Adding PATH export to $bashrc_file"
        echo "$path_export" >> "$bashrc_file"
        print_success "PATH export added to $bashrc_file"
    fi
}

# Function to create symlink safely
create_symlink() {
    local link_path=$1
    local target_path=$2
    
    print_info "Processing symlink: $link_path -> $target_path"
    
    # Check if target directory exists
    if [[ ! -d "$target_path" ]]; then
        print_error "Target directory $target_path does not exist. Skipping symlink creation."
        return 1
    fi
    
    # Check if symlink already exists and points to correct target
    if [[ -L "$link_path" ]]; then
        local current_target
        current_target=$(readlink "$link_path")
        if [[ "$current_target" == "$target_path" ]]; then
            print_success "Symlink $link_path already exists and points to correct target"
            return 0
        else
            print_info "Symlink $link_path exists but points to wrong target ($current_target). Removing and recreating."
            rm "$link_path"
        fi
    elif [[ -e "$link_path" ]]; then
        print_error "Path $link_path exists but is not a symlink. Manual intervention required."
        return 1
    fi
    
    # Create the symlink
    ln -s "$target_path" "$link_path"
    print_success "Created symlink: $link_path -> $target_path"
}

# Function to setup symlinks
setup_symlinks() {
    print_info "Setting up symbolic links..."
    
    # Define symlinks to create
    declare -A symlinks=(
        ["/home/ubuntu/.vscode-server"]="/home/ubuntu/dev/.vscode-server/"
        ["/home/ubuntu/.pixi"]="/home/ubuntu/dev/.pixi/"
    )
    
    local success_count=0
    local total_count=${#symlinks[@]}
    
    for link_path in "${!symlinks[@]}"; do
        target_path="${symlinks[$link_path]}"
        if create_symlink "$link_path" "$target_path"; then
            ((success_count++))
        fi
    done
    
    print_info "Symlink setup complete: $success_count/$total_count successful"
}

# Main execution
main() {
    print_info "Starting lambda.sh setup script..."
    print_info "This script will:"
    print_info "  1. Add PATH export to ~/.bashrc if not present"
    print_info "  2. Create symlinks for .vscode-server and .pixi"
    echo
    
    # Setup PATH export
    setup_path_export
    echo
    
    # Setup symlinks
    setup_symlinks
    echo
    
    print_success "lambda.sh setup script completed successfully!"
    print_info "Note: You may need to restart your shell or run 'source ~/.bashrc' for PATH changes to take effect."
}

# Run main function
main "$@"
