from flask import Flask, Response, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Set a non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import numpy as np
import base64
import requests
import math
import sqlite3
import io
import joblib
from collections import deque
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_fixed
import google.generativeai as genai
import faiss

gaze_evaluations = deque(maxlen=100)

app = Flask(__name__)

API_KEY = "NlN9R7AqnoTFwMsiIYI0"
GAZE_DETECTION_URL = "http://127.0.0.1:9001/gaze/gaze_detection?api_key=" + API_KEY
DISTANCE_TO_OBJECT = 1000  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm

GOOGLE_API_KEY="AIzaSyD8DkezpKW5QpI0QR4BzrT-QmHTmFY5NoU"
genai.configure(api_key=GOOGLE_API_KEY)

# Define the embedding function class
class GeminiEmbeddingFunction:
    def __init__(self):
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_mode = False
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def __call__(self, input):
        # Generate embeddings for the given input text
        embeddings = self.model.encode(input, convert_to_tensor=True)
        return embeddings.cpu().numpy()  # Convert tensor to numpy array

# Initialize the embedding function
embed_fn = GeminiEmbeddingFunction()

# Example documents (replace with your actual documents)
def load_documents(file_path):
    with open(file_path, 'r') as file:
        return file.read().split(';')  # Split documents by double newlines

documents = load_documents('chat_documents.txt')

# Initialize FAISS index
dimension = 384  # Dimensionality of the sentence embeddings, depends on the model used (384 for 'all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(dimension)  # Using L2 distance metric

# Embed the documents and add them to the FAISS index
embeddings = embed_fn(documents)  # Generate embeddings for the documents
index.add(embeddings.astype(np.float32))  # FAISS requires numpy array in float32 format

chat_model = genai.GenerativeModel("gemini-1.5-flash-latest")
embed_fn.document_mode = False

@app.route('/query', methods=['POST'])
def query():
    query_data = request.json
    query_text = query_data.get('query')

    if not query_text:
        return jsonify({'error': 'No query provided'}), 400

    query_embedding = embed_fn([query_text])
    query_embedding = query_embedding.astype(np.float32)
    query_embedding = query_embedding.reshape(1, -1)

    D, I = index.search(query_embedding, k=1)

    passage = documents[I[0][0]]

    prompt = f"""You are a psychiatrist specialized in the diagnosing ADHD. Explain symptoms, the severity or frequency of the symptoms, aspects of the ADHD condition, conditions that resemble ADHD, and the likelihood of the promper having ADHD. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    QUESTION: {query_text}
    PASSAGE: {passage}
    """

    print(query_text)

    answer = chat_model.generate_content(prompt)
    print(answer.text)
    return jsonify({"answer": answer.text})

def init_db():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS data (
                        id INTEGER PRIMARY KEY,
                        value REAL,
                        timestamp DATETIME,
                        data_type TEXT)''')
    conn.commit()
    conn.close()

def clear_database():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM data')
    conn.commit()
    conn.close()

@app.route('/')
def main_page():
    return render_template('page.html')

times_looking = 5
times_not_looking = 10

def calculate_average_score(db_path='data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT value FROM data WHERE data_type = 'score'")

    scores = cursor.fetchall()

    conn.close()

    if scores:
        score_values = [score[0] for score in scores]
        average_score = sum(score_values) / len(score_values)
        return average_score
    else:
        return None

model = joblib.load('adhd_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        average_score = calculate_average_score()
        std_input = calculate_std()
        focus_percentage = times_looking/(times_looking+times_not_looking)
    except KeyError:
        return jsonify({"error": "Missing required fields"}), 400

    features = np.array([[float(average_score), float(std_input), float(focus_percentage)]])
    prediction = model.predict(features)

    response = {
        "adhd_likelihood": prediction[0],
        "average_score": average_score,
        "std_input": std_input,
        "focus_percentage": focus_percentage
    }

    return jsonify(response)

import sqlite3
import math
from flask import jsonify

def calculate_std():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    cursor.execute("""
        WITH TimeDifferences AS (
            SELECT 
                timestamp,
                julianday(timestamp) - julianday(LAG(timestamp) OVER (ORDER BY timestamp)) AS time_diff
            FROM data
            WHERE data_type = 'player_input'
            ORDER BY timestamp DESC
            LIMIT 2000
        )
        SELECT time_diff FROM TimeDifferences WHERE time_diff IS NOT NULL
    """)
    
    # Fetch all time differences
    time_differences = cursor.fetchall()
    conn.close()

    if not time_differences:
        return jsonify({"error": "No player input data to calculate standard deviation."}), 400
    
    # Convert the time differences from days to seconds
    time_diffs_in_seconds = [td[0] * 86400 for td in time_differences]  # 1 day = 86400 seconds
    
    # Calculate the mean and standard deviation
    mean = sum(time_diffs_in_seconds) / len(time_diffs_in_seconds)
    variance = sum((x - mean) ** 2 for x in time_diffs_in_seconds) / len(time_diffs_in_seconds)
    std_dev = math.sqrt(variance)

    # Normalize the standard deviation using the mean
    if mean != 0:
        normalized_std_dev = std_dev / mean
    else:
        normalized_std_dev = 0  # If mean is 0, return 0 (to avoid division by zero)
    
    # Optionally, you could normalize using range (min-max normalization)
    # normalized_std_dev = std_dev / (max(time_diffs_in_seconds) - min(time_diffs_in_seconds))
    
    return normalized_std_dev


@app.route('/graph/pie_chart')
def pie_chart():
    global times_looking, times_not_looking
    return jsonify({
        "labels": ["Looking", "Not Looking"],
        "data": [times_looking, times_not_looking]
    })



@app.route('/submit-data', methods=['POST'])
def submit_data():
    data = request.get_json()  # Retrieve JSON data
    #print(f"Received data: {data}")

    value = data.get('value')
    data_type = data.get('data_type')
    timestamp = data.get('timestamp')  # Get the timestamp from the request
    
    if value is None or data_type is None or timestamp is None:
        return jsonify({"error": "value, data_type, and timestamp are required"}), 400
    
    # Ensure the timestamp is in the right format (optional: use datetime parsing)
    # try:
    #     parsed_timestamp = datetime.fromisoformat(timestamp)
    # except ValueError:
    #     return jsonify({"error": "Invalid timestamp format"}), 400

    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO data (value, timestamp, data_type) VALUES (?, ?, ?)', (value, timestamp, data_type))
    conn.commit()
    conn.close()

    return jsonify() 

def detect_gazes(frame: np.ndarray):
    img_encode = cv2.imencode(".jpg", frame)[1]
    img_base64 = base64.b64encode(img_encode)
    resp = requests.post(
        GAZE_DETECTION_URL,
        json={
            "api_key": API_KEY,
            "image": {"type": "base64", "value": img_base64.decode("utf-8")},
        },
    )
    gazes = resp.json()[0]["predictions"]
    return gazes

def draw_gaze(img: np.ndarray, gaze: dict):
    face = gaze["face"]
    x_min = int(face["x"] - face["width"] / 2)
    x_max = int(face["x"] + face["width"] / 2)
    y_min = int(face["y"] - face["height"] / 2)
    y_max = int(face["y"] + face["height"] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    _, imgW = img.shape[:2]
    arrow_length = imgW / 2
    dx = -arrow_length * np.sin(gaze["yaw"]) * np.cos(gaze["pitch"])
    dy = -arrow_length * np.sin(gaze["pitch"])
    cv2.arrowedLine(
        img,
        (int(face["x"]), int(face["y"])),
        (int(face["x"] + dx), int(face["y"] + dy)),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )

    for keypoint in face["landmarks"]:
        color, thickness, radius = (0, 255, 0), 2, 2
        x, y = int(keypoint["x"]), int(keypoint["y"])
        cv2.circle(img, (x, y), thickness, color, radius)

    label = "yaw {:.2f}  pitch {:.2f}".format(
        gaze["yaw"] / np.pi * 180, gaze["pitch"] / np.pi * 180
    )
    cv2.putText(
        img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    return img


def get_nearest_bird_position(db_path='data.db'):
    # Get the current timestamp in ISO 8601 format
    current_timestamp = datetime.now().isoformat()

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query all entries from the database
    cursor.execute('SELECT timestamp, value FROM data WHERE data_type = ? ORDER BY timestamp', ('bird_position',))
    rows = cursor.fetchall()

    conn.close()

    if not rows:
        return None  # Return None if the database is empty

    # Find the entry with the closest timestamp
    closest_position = None
    closest_time_diff = None

    for row in rows:
        db_timestamp = row[0]  # ISO 8601 timestamp from the database
        bird_position = row[1]

        # Parse timestamps to datetime objects
        db_time_obj = datetime.fromisoformat(db_timestamp)
        current_time_obj = datetime.fromisoformat(current_timestamp)

        # Calculate the time difference
        time_diff = abs((current_time_obj - db_time_obj).total_seconds())

        # Update the closest entry if this one is closer
        if closest_time_diff is None or time_diff < closest_time_diff:
            closest_time_diff = time_diff
            closest_position = bird_position

    return closest_position

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def generate():
    cap = cv2.VideoCapture(0)

    global times_looking, times_not_looking

    while True:
        _, frame = cap.read()
        gazes = detect_gazes(frame)

        if len(gazes) == 0:
            continue

        gaze = gazes[0]
        frame = draw_gaze(frame, gaze)

        max_pitch, min_pitch = -10, -30
        max_height, min_height = 550, 280

        position = get_nearest_bird_position( )
        bird_position = clamp((float(position.split(",")[1])-min_height)/(max_height-min_height), 0, 1)
        gaze_position = clamp(1 - ((gaze["pitch"] / np.pi * 180)-min_pitch)/(max_pitch-min_pitch), 0, 1)

        # Evaluate if the gaze is looking or not looking
        if abs(bird_position - gaze_position) <= 0.2:
            gaze_evaluations.append('looking')
        else:
            gaze_evaluations.append('not_looking')

        # print("GAZEPOSITION:  " + str(gaze_position))
        # print("BIRDPOSITION:  " + str(bird_position))
        # print()
        # Update the times_looking and times_not_looking based on the evaluations in the last 100 items
        times_looking = gaze_evaluations.count('looking')
        times_not_looking = gaze_evaluations.count('not_looking')

        # Convert frame to JPEG for streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/graph/<data_type>', methods=['GET'])
# def graph(data_type):
#     return Response(generate_graph(data_type), mimetype='image/png')

# @app.route('/graph/<data_type>')
# def graph(data_type):
#     img_data = generate_graph(data_type)
#     if not img_data:
#         return "No data found for this type", 404

#     # Decode the base64 image data back to binary and return it as an image response
#     img_bytes = base64.b64decode(img_data)
#     return Response(img_bytes, mimetype='image/png')

@app.route('/graph/<data_type>')
def graph(data_type):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    
    # Fetch the last 5 rows, ordered by timestamp
    cursor.execute('SELECT timestamp, value FROM data WHERE data_type = ? ORDER BY timestamp DESC LIMIT 30', (data_type,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No data found for this type", 404

    # Reverse the rows to get the correct order (oldest to newest)
    rows.reverse()

    # Extract and format timestamps to MM:SS
    formatted_timestamps = []
    values = []
    for row in rows:
        try:
            timestamp = row[0]  # Example: '2024-12-17T22:26:23.822453'
            time_part = timestamp.split('T')[1]  # Get '22:26:23.822453'
            minutes, seconds = time_part.split(':')[1:3]  # Extract minutes and seconds
            formatted_time = f"{minutes}:{seconds[:2]}"  # Keep only MM:SS
            formatted_timestamps.append(formatted_time)
            values.append(row[1])
        except (IndexError, ValueError) as e:
            print(f"Error parsing timestamp: {row[0]} -> {e}")

    # Prepare the data to be sent to Chart.js
    chart_data = {
        "labels": formatted_timestamps,  # Formatted X-axis labels
        "data": values  # Corresponding values
    }

    return jsonify(chart_data)  # Return data as JSON

if __name__ == "__main__":
    init_db()
    clear_database()
    app.run(debug=True)
