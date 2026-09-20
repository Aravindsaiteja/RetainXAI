@echo off
echo ============================================================
echo   Launching Explainable AI Streamlit Application
echo ============================================================

python -m streamlit run app/app.py --server.port 8501
pause
