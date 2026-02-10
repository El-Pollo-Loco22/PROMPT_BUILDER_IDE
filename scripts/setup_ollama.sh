#!/bin/bash
set -e

MODEL="${1:-llama3:8b}"

echo "=== Agentic Prompt IDE — Ollama Setup ==="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed."
    echo ""
    echo "Install it with:"
    echo "  macOS:  brew install ollama"
    echo "  Linux:  curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo "Ollama found: $(ollama --version)"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# Pull the model
echo "Pulling model: $MODEL"
ollama pull "$MODEL"

echo ""
echo "Setup complete! Model '$MODEL' is ready."
echo "Ollama is running at http://localhost:11434"
