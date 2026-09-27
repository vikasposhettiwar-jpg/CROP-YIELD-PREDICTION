# Smart Crop Yield Prediction and Recommendation System

Now organized as a clear **frontend + backend** project.

## Architecture
- `backend/` - Flask REST API + ML pipeline
- `frontend/` - standalone HTML/CSS/JS client

## Backend
- `backend/app.py` - API server
- `backend/model/train_model.py` - model training and auto model selection
- `backend/model/predict.py` - prediction/recommendation helpers
- `backend/utils/preprocessing.py` - data cleaning, encoding, scaling
- `backend/data/crop_yield_india.csv` - India-focused sample dataset
- `backend/artifacts/` - saved model + JSON artifacts

### Backend APIs
- `GET /api/health`
- `GET /api/dashboard`
- `POST /api/predict`
- `POST /api/recommend`

## Frontend
- `frontend/index.html`
- `frontend/css/style.css`
- `frontend/js/app.js`
- `frontend/images/` (generated model charts)

Frontend calls backend at `http://127.0.0.1:5000`.

## Run Instructions
1. Create venv:
   - `python -m venv .venv`
   - `.venv\Scripts\Activate.ps1`
2. Install backend deps:
   - `pip install -r backend/requirements.txt`
3. Train model:
   - `python backend/model/train_model.py`
4. Start backend API:
   - `python backend/app.py`
5. Open frontend:
   - Open `frontend/index.html` in browser
   - Or run: `python -m http.server 5500` and open `http://127.0.0.1:5500/frontend/`

## ML Details
- Models: Random Forest, Gradient Boosting, Linear Regression
- Metrics: RMSE, MAE, R2
- Auto selects best model and saves with joblib
- Feature importance generated for tree-based model

