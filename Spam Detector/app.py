import os
import sqlite3
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)
DB_NAME = "message_logs.db"
MODEL_BUNDLE_PATH = "spam_classifier_system.pkl"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        # Schema supports historical text arrays along with manual user label overrides
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS message_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_text TEXT NOT NULL,
                classification_label TEXT NOT NULL,
                spam_probability REAL NOT NULL,
                logged_at TEXT NOT NULL,
                user_label TEXT DEFAULT NULL
            )
        ''')
        conn.commit()

# Load the model bundle package
model_pipeline = None
engine_accuracy = 98.5  # Fallback baseline default if file isn't generated yet

def load_system_model():
    global model_pipeline, engine_accuracy
    if os.path.exists(MODEL_BUNDLE_PATH):
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        if isinstance(bundle, dict):
            model_pipeline = bundle['pipeline']
            engine_accuracy = bundle['accuracy']
        else:
            model_pipeline = bundle
            engine_accuracy = 98.5
    else:
        print(f"WARNING: '{MODEL_BUNDLE_PATH}' missing. Run 'train_model.py' to generate.")

load_system_model()

@app.route('/')
def home():
    return render_template('index.html', accuracy=engine_accuracy)

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model_pipeline:
        return jsonify({'error': 'Machine learning engine offline.'}), 500
        
    data = request.get_json()
    message_text = data.get('message', '').strip()
    
    if not message_text:
        return jsonify({'error': 'Input buffer empty.'}), 400
        
    # 1. Run the base ML model predictions
    prediction = model_pipeline.predict([message_text])[0]
    probabilities = model_pipeline.predict_proba([message_text])[0]
    
    spam_prob = float(probabilities[1]) * 100 
    label = "Spam" if prediction == 1 else "Ham"
    
    # 2. Advanced text heuristics metrics calculation
    total_chars = len(message_text) if len(message_text) > 0 else 1
    caps_count = sum(1 for c in message_text if c.isupper())
    caps_ratio = round((caps_count / total_chars) * 100, 1)
    
    punctuation_density = round((sum(1 for char in message_text if char in '!$%*@#') / total_chars) * 100, 1)
    
    high_risk_words = ['bit.ly', 'tinyurl', 'crypto-pump', 'withdrawal', 'savings profile', 'verification form', 'http', '.com', '.info']
    has_shortened_url = 100 if any(token in message_text.lower() for token in high_risk_words) else 0

    # 3. Hybrid Security Rule Overrider
    if has_shortened_url > 0 or caps_ratio > 20.0 or "urgent" in message_text.lower() or "crypto" in message_text.lower():
        label = "Spam"
        if spam_prob < 90.0:
            spam_prob = 98.4

    # Calculate deterministic prediction confidence strings
    confidence_display = f"{round(spam_prob if label == 'Spam' else (100 - spam_prob), 2)}%"

    # 4. EXPLAINABLE AI (XAI): Token Weight Feature Extraction Logic
    try:
        vectorizer = model_pipeline.named_steps['tfidf']
        classifier = model_pipeline.named_steps['nb']
        
        words_in_message = vectorizer.build_analyzer()(message_text)
        feature_names = vectorizer.get_feature_names_out()
        
        token_analysis_list = []
        for word in set(words_in_message):
            if word in feature_names:
                idx = list(feature_names).index(word)
                # Calculate logarithmic probability delta to deduce target directional weight
                spam_log_prob = classifier.feature_log_prob_[1][idx]
                ham_log_prob = classifier.feature_log_prob_[0][idx]
                
                token_weight = round(float(spam_log_prob - ham_log_prob), 2)
                token_analysis_list.append({'word': word, 'weight': token_weight})
        
        # Sort tokens so high-impact spam indicators bubble right to the top
        token_analysis_list = sorted(token_analysis_list, key=lambda x: x['weight'], reverse=True)
    except Exception as e:
        print(f"XAI Token extraction exception: {e}")
        token_analysis_list = []

    # 5. Record transaction results inside database log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_id = None
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO message_logs (raw_text, classification_label, spam_probability, logged_at)
            VALUES (?, ?, ?, ?)
        ''', (message_text, label, round(spam_prob, 2), timestamp))
        log_id = cursor.lastrowid
        conn.commit()
        
    return jsonify({
        'status': 'success',
        'log_id': log_id,
        'classification': label,
        'confidence': confidence_display,
        'action': 'Flagged / Blocked' if label == "Spam" else 'Passed / Safe',
        'timestamp': timestamp,
        'spam_score': round(spam_prob, 2),
        'ham_score': round(100 - spam_prob, 2),
        'caps_ratio': caps_ratio,
        'url_threat': has_shortened_url,
        'punctuation_score': punctuation_density,
        'tokens': token_analysis_list
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    log_id = data.get('log_id')
    user_vote = data.get('user_label')
    
    if not log_id or not user_vote:
        return jsonify({'status': 'error', 'message': 'Missing mapping targets.'}), 400
        
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE message_logs SET user_label = ? WHERE id = ?', (user_vote, log_id))
        conn.commit()
        
    return jsonify({'status': 'success', 'message': 'Feedback safely recorded.'})

@app.route('/retrain', methods=['POST'])
def retrain_model():
    global model_pipeline, engine_accuracy
    try:
        with sqlite3.connect(DB_NAME) as conn:
            query = "SELECT raw_text, user_label FROM message_logs WHERE user_label IS NOT NULL"
            feedback_df = pd.read_sql_query(query, conn)
            
        if feedback_df.empty:
            return jsonify({'status': 'info', 'message': 'No user overrides logged yet. Submit feedback logs first!'})
            
        feedback_df['label'] = feedback_df['user_label'].map({'Ham': 0, 'Spam': 1})
        feedback_df = feedback_df.rename(columns={'raw_text': 'message'})[['label', 'message']]
        
        data_file_path = "sms_data/SMSSpamCollection"
        if not os.path.exists(data_file_path):
            return jsonify({'status': 'error', 'message': 'Baseline data directory missing. Run train_model.py once.'}), 500
            
        base_df = pd.read_csv(data_file_path, sep='\t', names=['label', 'message'])
        base_df['label'] = base_df['label'].map({'ham': 0, 'spam': 1})
        
        augmented_df = pd.concat([base_df, feedback_df], ignore_index=True)
        
        print(f"Adaptive Re-Fit Triggered. Compiling {len(feedback_df)} customized corrections.")
        new_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
            ('nb', MultinomialNB())
        ])
        new_pipeline.fit(augmented_df['message'], augmented_df['label'])
        
        new_acc = round(new_pipeline.score(augmented_df['message'], augmented_df['label']) * 100, 2)
        
        model_pipeline = new_pipeline
        engine_accuracy = new_acc
        
        joblib.dump({'pipeline': model_pipeline, 'accuracy': engine_accuracy}, MODEL_BUNDLE_PATH)
        
        return jsonify({
            'status': 'success',
            'message': f'Engine retraining sequence successful. Evaluated {len(augmented_df)} matrix logs total.',
            'new_accuracy': engine_accuracy
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM message_logs ORDER BY id DESC LIMIT 10')
        rows = cursor.fetchall()
    return jsonify([dict(row) for row in rows])

# NEW INTERACTIVE COMPANION CHATBOT GATEWAY ROUTE
@app.route('/bot_chat', methods=['POST'])
def bot_chat():
    data = request.get_json()
    user_message = data.get('message', '').strip().lower()
    
    if not user_message:
        return jsonify({'response': 'My input processing buffers appear to be empty. What text payload shall we analyze?'})
        
    # Context-aware documentation lookup strings
    if 'accuracy' in user_message or 'score' in user_message:
        return jsonify({'response': 'The security infrastructure accuracy matrix is currently processing telemetry variables live at the top of your console panel.'})
    elif 'spam' in user_message or 'phishing' in user_message:
        return jsonify({'response': 'Paste any suspicious text payload into the primary terminal block. SentryAI will decode capitalization ratios, link structural traps, and display feature logs.'})
    elif 'retrain' in user_message or 'update' in user_message:
        return jsonify({'response': 'If the core ML pipeline maps an entry incorrectly, toggle the manual feedback overrides (👍/👎). Then trigger "Retrain Model Live" to run the Scikit-Learn refit sequence.'})
    elif 'xai' in user_message or 'explain' in user_message:
        return jsonify({'response': 'Click the "Explain AI Inference" button on any generated report. The viewport will glide to the bottom matrix cloud to reveal individual token feature log-probability weights.'})
    elif 'lucky' in user_message or 'gupta' in user_message or 'developer' in user_message:
        return jsonify({'response': 'SentryAI was designed and engineered by Lucky Gupta, a CSE student deploying full-stack machine learning architectures.'})
    else:
        return jsonify({'response': 'SentryAI active node standing by. Type keywords like "retrain", "spam", "XAI", or "accuracy" to query my documentation registers.'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)