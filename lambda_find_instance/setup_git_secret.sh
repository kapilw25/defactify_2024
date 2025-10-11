#!/bin/bash

# Setup script for git-secret encryption

echo "=== Lambda Labs Instance Finder - Secret Setup ==="

# Check if git-secret is installed
if ! command -v git-secret &> /dev/null; then
    echo "❌ git-secret not found. Installing..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git-secret
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install git-secret
    fi
fi

# Initialize git if needed
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# Initialize git-secret
if [ ! -d .gitsecret ]; then
    echo "Initializing git-secret..."
    git-secret init
fi

# Add .env file to git-secret tracking
if [ -f config/.env ]; then
    echo "Adding config/.env to git-secret..."
    git-secret add config/.env
    git-secret hide
    echo "✅ Secrets encrypted! File: config/.env.secret"
    echo ""
    echo "To decrypt: git-secret reveal"
    echo "To encrypt after changes: git-secret hide"
else
    echo "❌ config/.env not found. Create it first."
fi

echo ""
echo "=== Setup Complete ==="
echo "Remember: Never commit config/.env (it's in .gitignore)"
echo "Only commit config/.env.secret"
