# speed-backend

This is a FastAPI application that predicts the next speed based on historical speed data using a machine learning model.

## Deployment

This application is deployed on Render. The deployment is configured through the `render.yaml` file in this repository.

## Local Development

To run this application locally:

1. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   uvicorn app:app --reload --port 10000
   ```

4. Access the API documentation at http://127.0.0.1:10000/docs

## API Endpoints

- POST `/predict-speed` - Predicts the next speed based on input data

The API expects a JSON payload with the following structure:

```json
{
  "speeds": [10, 20, 30, 40]
}
```

Where `speeds` is an array of numerical values representing historical speed data.
