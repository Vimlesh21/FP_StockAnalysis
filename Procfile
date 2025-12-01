web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
api: uvicorn src.api.predict_simple:app --host 0.0.0.0 --port=$PORT --workers 1
