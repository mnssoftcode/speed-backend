# speed-backend

This is a FastAPI application that predicts the next speed based on historical speed data and detects potential crashes using machine learning models.

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

1. POST `/predict-speed` - Predicts the next speed based on input data

   The API expects a JSON payload with the following structure:

   ```json
   {
     "speeds": [10, 20, 30, 40]
   }
   ```

   Where `speeds` is an array of numerical values representing historical speed data.

2. POST `/predict-crash` - Predicts if a crash will occur based on driving parameters

   The API expects a JSON payload with the following structure:

   ```json
   {
     "speed": 60.5,
     "accel": 2.3,
     "gyro": 0.1,
     "jerk": 0.05
   }
   ```

   Where:
   - `speed` is the current speed
   - `accel` is the acceleration
   - `gyro` is the gyroscope reading
   - `jerk` is the jerk (rate of change of acceleration)

3. POST `/predict-risk` - Predicts the risk level based on driving parameters

   The API expects a JSON payload with the following structure:

   ```json
   {
     "speed": 60.5,
     "accel": 2.3,
     "brake": 1.2,
     "gyro": 0.1,
     "jerk": 0.05
   }
   ```

   Where:
   - `speed` is the current speed
   - `accel` is the acceleration
   - `brake` is the brake pressure
   - `gyro` is the gyroscope reading
   - `jerk` is the jerk (rate of change of acceleration)

   The response will include a risk level (0-2) and a corresponding status:
   - 0: LOW RISK
   - 1: MEDIUM RISK
   - 2: HIGH RISK

4. POST `/predict-risk-rf` - Predicts the risk level using a Random Forest model

   The API expects the same JSON payload as the regular risk prediction endpoint:

   ```json
   {
     "speed": 60.5,
     "accel": 2.3,
     "brake": 1.2,
     "gyro": 0.1,
     "jerk": 0.05
   }
   ```

   The response format is identical to the regular risk prediction endpoint, with a risk level (0-2) and corresponding status.
