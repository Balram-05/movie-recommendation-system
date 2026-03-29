# 🎬 Movie Recommendation System

### _A Full-Stack Weekend Project_

A production-ready recommendation engine built from scratch over a weekend. This project demonstrates a complete ML lifecycle: from data cleaning and feature engineering to a containerized microservices deployment.

## 🛠️ Technology Stack

- **Machine Learning:** Python, Scikit-Learn, Pandas, NumPy (KNN + Cosine Similarity)
- **API (Backend):** FastAPI (Uvicorn)
- **UI (Frontend):** Streamlit
- **DevOps:** Docker, Docker Compose
- **Cloud Hosting:** AWS EC2 (Ubuntu 24.04 LTS), Render

## 🏗️ Architecture & Deployment

This project supports two deployment modes:

1. **Microservices (AWS/Docker):** The `backend/` (FastAPI) and `frontend/` (Streamlit) run as separate Docker containers, communicating over a virtual network.
2. **Standalone (Render):** A unified `render_app.py` that merges logic and UI into a single process to fit within free-tier RAM limits (512MB).

## 🚀 Future Improvements

- [ ] **Hybrid Filtering:** Combine content-based results with Collaborative Filtering (User Ratings).
- [ ] **NLP Enhancement:** Use Word2Vec or BERT embeddings instead of CountVectorizer for deeper semantic matching.
- [ ] **CI/CD:** Implement GitHub Actions to automatically deploy to Render on every push.
