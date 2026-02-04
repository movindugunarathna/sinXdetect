# SinXdetect - Sinhala Human vs AI Text Classifier

🌐 **Live Demo**: [https://sinxdetect.movindu.com/](https://sinxdetect.movindu.com/)

A web application that classifies Sinhala text as **Human-written** or **AI-generated** using deep learning models with LIME-based explainability.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
  - [Local Development](#local-development)
  - [Docker Setup](#docker-setup)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

## 🎯 Overview

SinXdetect is a full-stack application designed to detect AI-generated Sinhala text. It uses a fine-tuned SinBERT model for classification and provides interpretable results using LIME (Local Interpretable Model-agnostic Explanations).

## ✨ Features

- **Binary Classification**: Classifies text as HUMAN or AI-generated
- **Batch Processing**: Classify multiple texts efficiently
- **Explainability**: LIME-based word importance highlighting
- **Modern UI**: Clean, responsive React frontend with Tailwind CSS
- **RESTful API**: FastAPI backend with automatic documentation
- **Docker Support**: Easy deployment with Docker Compose

## 📁 Project Structure

```
sinXdetect/
├── Dockerfile               # Unified multi-stage Dockerfile (frontend + backend)
├── docker-compose.yml       # Production Docker config (unified container)
├── docker-compose.dev.yml   # Development Docker config (separate services)
├── nginx.conf               # Nginx configuration for production
├── supervisord.conf         # Process manager for production
├── start-docker.sh          # Linux/Mac startup script
├── start-docker.bat         # Windows startup script
├── stop-docker.sh           # Linux/Mac stop script
├── stop-docker.bat          # Windows stop script
├── backend/                 # FastAPI backend
│   ├── app.py              # Main API application
│   ├── classify_text.py    # Classification logic
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Backend-only container (for dev mode)
├── frontend/               # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   └── main.jsx       # Entry point
│   ├── package.json       # Node dependencies
│   ├── Dockerfile         # Frontend production container
│   └── Dockerfile.dev     # Frontend dev container
└── ml/                     # Machine learning models
    ├── models/            # Trained models
    │   └── sinbert_sinhala_classifier/
    └── *.ipynb           # Training notebooks
```

## 📋 Prerequisites

### For Local Development

- Python 3.11+
- Node.js 18+
- npm or yarn
- **Git LFS** (for pulling trained ML models) - [Installation Guide](https://git-lfs.github.com/)

### Docker Deployment

- Docker Engine 20.10+
- Docker Compose 1.29+
- **Git LFS** (for pulling trained ML models)
- At least 4GB free RAM (for ML model loading)
- At least 6GB free disk space

## 🚀 Installation

### Local Development

#### 1. Clone the Repository

```bash
git clone https://github.com/movindugunarathna/sinXdetect.git
cd sinXdetect
```

> ⚠️ **Important**: The trained ML models are stored using **Git LFS** (Large File Storage). To pull the models, you need to install Git LFS first:
>
> ```bash
> # Install Git LFS (one-time setup)
> git lfs install
>
> # Pull the model files
> git lfs pull
> ```
>
> Without Git LFS, the model files will be placeholder pointers and the application won't work.

#### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python app.py
# Or with uvicorn:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The backend API will be available at: `http://localhost:8000`

#### 3. Frontend Setup

```bash
# Open a new terminal and navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment file (for development)
echo "VITE_API_URL=http://localhost:8000" > .env

# Start development server
npm run dev
```

The frontend will be available at: `http://localhost:5173`

### Docker Setup

The project uses a **unified multi-stage Dockerfile** that combines both frontend and backend into a single container for production deployment. This simplifies deployment and ensures consistency.

#### Environment Overview

| Environment     | Command                   | Frontend URL                   | Backend API URL                    |
| --------------- | ------------------------- | ------------------------------ | ---------------------------------- |
| **Development** | `./start-docker.sh dev`   | http://localhost:5173          | http://localhost:8000              |
| **Local**       | `./start-docker.sh local` | http://localhost:3000          | http://localhost:3000/api          |
| **Production**  | `./start-docker.sh prod`  | https://sinxdetect.movindu.com | https://api.sinxdetect.movindu.com |

#### Architecture

- **Development Mode**: Separate containers for backend and frontend with hot-reload support
- **Local Mode**: Single container for local testing (port 3000)
- **Production Mode**: Optimized single container for deployment (port 80)

#### Quick Start

**Development with Hot-Reload (Recommended for coding):**

```bash
# Linux/Mac/Git Bash
./start-docker.sh dev

# Windows Command Prompt
start-docker.bat dev
```

**Local Testing (Combined container):**

```bash
# Linux/Mac/Git Bash
./start-docker.sh local

# Windows Command Prompt
start-docker.bat local
```

**Production Deployment:**

```bash
# Linux/Mac/Git Bash
./start-docker.sh prod

# Windows Command Prompt
start-docker.bat prod
```

> 💡 **Windows Users**: If using Git Bash, use the `.sh` scripts. The `.bat` files only work in Command Prompt or PowerShell.

> ⏳ **Note**: First startup may take 2-3 minutes while the ML model loads into memory.

#### Stopping Services

```bash
# Stop all services
./stop-docker.sh

# Stop specific environment
./stop-docker.sh dev
./stop-docker.sh prod

# Stop and remove volumes
./stop-docker.sh clean
```

#### Manual Docker Commands

```bash
# Development mode
docker compose -f docker-compose.dev.yml up --build --remove-orphans

# Local testing mode
docker compose up --build -d --remove-orphans

# Production mode
docker compose -f docker-compose.prod.yml up --build -d --remove-orphans

# Build the unified container
docker compose build

# Start in foreground (see logs)
docker compose up

# Start in background (detached)
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v

# Rebuild without cache
docker compose build --no-cache
```

#### Development Mode Commands

```bash
# Start development mode
docker compose -f docker-compose.dev.yml up --build

# Stop development services
docker compose -f docker-compose.dev.yml down
```

#### Stopping Docker Services

**Linux/Mac/Git Bash (Windows):**

```bash
./stop-docker.sh          # Stop services
./stop-docker.sh clean    # Stop and remove volumes
```

**Windows Command Prompt/PowerShell:**

```cmd
stop-docker.bat           REM Stop services
stop-docker.bat clean     REM Stop and remove volumes
```

## 📖 Usage

### Web Interface

1. Open the application in your browser
2. Enter or paste Sinhala text into the text area
3. Choose your action:
   - **Classify**: Quick classification (HUMAN/AI)
   - **Explain with LIME**: Detailed word-level analysis

### Example Text

```
මෙය කෘතිම බුද්ධි මගින් ලියන ලද වාක්‍යයකි
```

## 🔌 API Endpoints

| Endpoint          | Method | Description              |
| ----------------- | ------ | ------------------------ |
| `/`               | GET    | API information          |
| `/health`         | GET    | Health check             |
| `/classify`       | POST   | Classify single text     |
| `/classify-batch` | POST   | Classify multiple texts  |
| `/explain`        | POST   | Get LIME explanation     |
| `/docs`           | GET    | Swagger UI documentation |

### Example API Requests

**Classify Text:**

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "සිංහල පෙළ උදාහරණයක්", "return_probabilities": true}'
```

**Get LIME Explanation:**

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{"text": "සිංහල පෙළ උදාහරණයක්", "num_samples": 100}'
```

## ⚙️ Configuration

### Environment Variables

**Backend:**
| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to ML model | `ml/models/sinbert_sinhala_classifier` |
| `PYTHONUNBUFFERED` | Immediate stdout/stderr | `1` |

**Frontend:**
| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | Environment-specific (see below) |

**Environment-specific API URLs:**
| Environment | VITE_API_URL |
|-------------|--------------|
| Development | `http://localhost:8000` |
| Local | `http://localhost:3000/api` |
| Production | `https://api.sinxdetect.movindu.com` |

### Docker Configuration

Each environment has its own docker-compose file:

- `docker-compose.dev.yml` - Development with hot-reload
- `docker-compose.yml` - Local testing
- `docker-compose.prod.yml` - Production deployment

**Production URLs:**

- Frontend: `https://sinxdetect.movindu.com`
- Backend API: `https://api.sinxdetect.movindu.com`

Example customization in `docker-compose.yml`:

```yaml
services:
  sinxdetect:
    environment:
      - MODEL_PATH=/app/ml/models/sinbert_sinhala_classifier
    build:
      args:
        - VITE_API_URL=http://localhost:3000/api
```

## 🔧 Troubleshooting

### Backend Issues

**Model not loading:**

```bash
# Verify model directory structure
ls -la ml/models/sinbert_sinhala_classifier/
# Should contain: config.json, tf_model.h5, tokenizer_config.json, etc.
```

**Port already in use:**

```bash
# Change port in docker-compose.yml
ports:
  - '8001:8000'  # Use host port 8001
```

### Frontend Issues

**API connection error:**

- Ensure backend is running on port 8000
- Check `VITE_API_URL` in `.env` file

**Build failures:**

```bash
# Clear cache and rebuild
docker compose down -v
docker compose build --no-cache frontend
```

### Docker Issues

**Check service health:**

```bash
docker compose ps
docker compose logs backend
docker compose logs frontend
```

**Memory issues:**

- Ensure at least 4GB RAM available
- Backend with ML model requires significant memory

## 📊 Model Information

- **Model**: Fine-tuned SinBERT
- **Task**: Binary classification (HUMAN vs AI)
- **Explainability**: LIME text explainer
- **Training notebooks**: Available in `ml/` directory

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Created by**: Movindu Gunarathna
**Project**: Final Year Project (FYP)  
**Version**: 2.0.0
