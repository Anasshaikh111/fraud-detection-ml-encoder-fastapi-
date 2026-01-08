Write-Host "Cleaning Git repository..."

# Remove .venv from Git tracking (if exists)
if (Test-Path ".venv") {
    git rm -r --cached .venv
    Write-Host ".venv removed from Git tracking"
}

# Create .gitignore
Set-Content .gitignore @(
    ".venv/",
    "venv/",
    "__pycache__/",
    "*.pyc",
    "*.pkl",
    "*.joblib",
    "*.h5",
    "*.csv",
    ".DS_Store"
)

Write-Host ".gitignore created"

# Fix line endings warning
git config --global core.autocrlf true

# Commit changes
git add .
git commit -m "Clean repo: remove venv and ignore ML artifacts"

# Push to GitHub
git push origin main

Write-Host "Git cleanup & push completed!"
