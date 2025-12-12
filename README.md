## How it works
This model predicts Atlanta home values using
- **Historical price trends** (1, 3, 6 months ago)
- **Rolling averages** (3, 6 months)
- **Seasonal patterns** (month/year)

## How to run 
### Terminal 1
`streamlit run src/streamlit_app.py`

### Terminal 2
`python src/serve_fastapi.py`