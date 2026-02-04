# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2024

### Release

🌐 **Live URL**: [https://sinxdetect.movindu.com/](https://sinxdetect.movindu.com/)

### Added

- Full-stack web application for Sinhala text classification
- Binary classification (Human vs AI-generated text)
- LIME-based explainability for word importance highlighting
- Modern React frontend with Tailwind CSS
- FastAPI backend with RESTful API
- Docker support for easy deployment
- Batch processing for multiple texts
- Interactive API documentation (Swagger UI)

### Features

- **Text Classification**: Accurately classifies Sinhala text as HUMAN or AI-generated
- **Explainability**: LIME text explainer provides word-level analysis
- **Responsive UI**: Clean, modern interface built with React and Tailwind CSS
- **Production Ready**: Docker Compose setup for deployment
- **Development Mode**: Hot-reload support for active development

### Technical Stack

- **Frontend**: React 19, Vite, Tailwind CSS
- **Backend**: FastAPI, Python 3.11+
- **ML Model**: Fine-tuned SinBERT
- **Deployment**: Docker, Nginx, Supervisord
