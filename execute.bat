@echo off
echo ==============================================
echo      CKD PREDICTION SYSTEM LAUNCHER
echo ==============================================

echo.
echo [1/2] Starting FastAPI Backend on Port 8000...
:: Opens a new window titled "CKD_Backend_API" and runs uvicorn
start "CKD_Backend_API" cmd /k "python -m uvicorn app.api:app --reload --port 8000"

echo Waiting 5 seconds for backend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo [2/2] Starting Doctor Portal (Streamlit) on Port 8501...
:: Opens a new window titled "CKD_Doctor_Portal" and runs streamlit
start "Doctor Portal" python -m streamlit run app/dashboard.py"

echo.
echo ==============================================
echo      SYSTEM LAUNCHED SUCCESSFULLY
echo ==============================================
echo Backend:   http://localhost:8000/docs
echo Frontend:  http://localhost:8501
echo.
echo NOTE: Please keep the pop-up windows open. 
echo Close them when you are done.
echo.
pause
