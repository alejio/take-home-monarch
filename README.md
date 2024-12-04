# take-home-monarch

## Overview

Contains my solution to the take-home assignment for the Monarch Lead ML Eng position.

Specifically:
- The RFC can be found in `docs/RFC - Upgrading Transaction Enrichment.pdf`.
- The Streamlit app is `streamlit_eda.py` and deployed at https://streamlit-eda-1048447215780.europe-west2.run.app/


## Running the app locally

### Docker

1. Clone the repo
2. Run `docker build -t streamlit-eda .`
3. Run `docker run -p 8501:8501 streamlit-eda`
4. Open http://localhost:8501/ in your browser

### Python

1. Clone the repo
2. Create and activate your virtual environment
2. Run `pip install -r requirements.txt`
3. Run `streamlit run streamlit_eda.py`