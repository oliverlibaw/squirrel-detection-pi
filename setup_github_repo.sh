#!/bin/bash

# Setup script for GitHub repository
echo "🐿️ Setting up Squirrel Detection GitHub Repository"
echo "=================================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files to git
echo "📝 Adding files to git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Squirrel detection with YOLOV11s on Raspberry Pi

- Real-time squirrel detection using custom YOLOV11s model
- GUI and headless modes for different environments
- Optimized for Raspberry Pi with Hailo AI accelerator
- Comprehensive documentation and testing"

echo ""
echo "✅ Repository setup complete!"
echo ""
echo "🚀 Next steps:"
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Name it: squirrel-detection-pi"
echo "   - Make it public or private as preferred"
echo "   - Don't initialize with README (we already have one)"
echo ""
echo "2. Connect your local repository to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/squirrel-detection-pi.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Update the following files with your GitHub username:"
echo "   - README.md (replace 'yourusername' with your actual username)"
echo "   - setup.py (update author and URL)"
echo "   - CONTRIBUTING.md (update URLs)"
echo ""
echo "4. Push your changes:"
echo "   git add ."
echo "   git commit -m 'Update repository URLs and author information'"
echo "   git push"
echo ""
echo "🎉 Your squirrel detection project will be live on GitHub!" 