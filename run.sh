#!/bin/bash

# Ensure the virtual environment is activated if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the Streamlit application
streamlit run streamlit_app.py 