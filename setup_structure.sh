#!/bin/bash
# setup_structure.sh

# Root directory
mkdir -p trendops
cd trendops

# Source directories
mkdir -p src/trendops/trigger
mkdir -p src/trendops/collector
mkdir -p src/trendops/queue
mkdir -p src/trendops/config
mkdir -p src/trendops/utils

# Test directories
mkdir -p tests/trigger
mkdir -p tests/collector
mkdir -p tests/queue

# Docker
mkdir -p docker

# Logs
mkdir -p logs

# Create __init__.py files
touch src/trendops/__init__.py
touch src/trendops/trigger/__init__.py
touch src/trendops/collector/__init__.py
touch src/trendops/queue/__init__.py
touch src/trendops/config/__init__.py
touch src/trendops/utils/__init__.py
touch tests/__init__.py
touch tests/trigger/__init__.py
touch tests/collector/__init__.py
touch tests/queue/__init__.py

# Create placeholder files
touch src/trendops/trigger/trigger_google.py
touch src/trendops/trigger/trigger_scorer.py
touch src/trendops/collector/collector_base.py
touch src/trendops/collector/collector_rss_google.py
touch src/trendops/collector/collector_rss_naver.py
touch src/trendops/queue/queue_redis.py
touch src/trendops/queue/queue_models.py
touch src/trendops/config/settings.py
touch src/trendops/utils/logger.py

# Create .gitignore
cat > .gitignore << 'EOF'
# Byte-compiled
__pycache__/
*.py[cod]
*$py.class

# Environment
.env
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp

# Logs
logs/
*.log
*.jsonl

# Distribution
dist/
build/
*.egg-info/

# Testing
.pytest_cache/
.coverage
htmlcov/

# MyPy
.mypy_cache/

# Docker
docker/data/
EOF

echo "✅ TrendOps directory structure created successfully!"