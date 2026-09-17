@echo off
cd /d "%~dp0"

REM Activate the virtual environment where pypdf, python-docx, openpyxl etc. are installed
call venv\Scripts\activate.bat

REM Run the Streamlit app (change app.py if your main file has a different name)
streamlit run main.py

pause