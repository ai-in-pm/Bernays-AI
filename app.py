from flask import Flask, request, jsonify, render_template, render_template_string
from dotenv import load_dotenv
import os
import logging
import numpy as np
import json
import sounddevice as sd
from datetime import datetime
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.neural_oscillator import NeuralOscillator, BrainwaveStates
from modules.interaction_agent import InteractionAgent, InteractionState

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (BrainwaveStates, InteractionState)):
            return obj.name
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder

# Configure app
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET')

# Feature flags
FEATURES = {
    'neural_modulation': True,
    'visual_entrainment': True,
    'audio_synthesis': True
}

# Neural configuration
NEURAL_CONFIG = {
    'sample_rate': int(os.getenv('DEFAULT_SAMPLE_RATE', 44100)),
    'min_freq': float(os.getenv('MIN_FREQUENCY', 1.0)),
    'max_freq': float(os.getenv('MAX_FREQUENCY', 40.0)),
    'duration': float(os.getenv('DEFAULT_DURATION', 5.0))
}

# Initialize the agent
agent = InteractionAgent(NEURAL_CONFIG)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Write the template to a file
with open('templates/chat.html', 'w') as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Subliminal AI Agent Chat</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 20px auto;
            padding: 0 20px;
        }
        .chat-container {
            border: 1px solid #ddd;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            min-height: 400px;
            max-height: 400px;
            overflow-y: auto;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .agent {
            background-color: #f5f5f5;
        }
        .user {
            background-color: #e3f2fd;
            text-align: right;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .metrics-container {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .note {
            color: #dc3545;
            margin-top: 10px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>Subliminal AI Agent Chat</h1>
    
    <div class="chat-container" id="chatContainer"></div>
    
    <div class="input-container">
        <input type="text" id="userInput" placeholder="Type your message here..." />
    </div>
    
    <div class="metrics-container">
        <h2>Interaction Metrics</h2>
        <pre id="metricsDisplay"></pre>
    </div>
    
    <p class="note">Note: This is a research prototype. Neural modulation features are active.</p>

    <script>
        const chatContainer = document.getElementById('chatContainer');
        const userInput = document.getElementById('userInput');
        const metricsDisplay = document.getElementById('metricsDisplay');
        
        function addMessage(message, isUser) {
            const div = document.createElement('div');
            div.className = `message ${isUser ? 'user' : 'agent'}`;
            div.textContent = message;
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function updateMetrics(metrics) {
            if (!metrics) return;
            
            const formattedMetrics = {
                'Total Interactions': metrics.total_interactions,
                'Current State': metrics.current_state,
                'Influence Level': (metrics.influence_level * 100).toFixed(1) + '%',
                'Session Duration': formatDuration(metrics.session_duration_seconds)
            };
            
            metricsDisplay.textContent = JSON.stringify(formattedMetrics, null, 2);
        }
        
        function formatDuration(seconds) {
            if (seconds < 60) return `${seconds} seconds`;
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            return `${minutes}m ${remainingSeconds}s`;
        }
        
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && userInput.value.trim()) {
                const message = userInput.value.trim();
                addMessage(message, true);
                sendMessage(message);
                userInput.value = '';
            }
        });
        
        async function sendMessage(message) {
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage('Error: ' + data.error, false);
                    return;
                }
                
                if (data.response && data.response.text) {
                    addMessage(data.response.text, false);
                    
                    if (data.metrics) {
                        updateMetrics(data.metrics);
                    }
                    
                    // Play audio if available
                    if (data.response.audio) {
                        try {
                            const audio = new Audio('data:audio/wav;base64,' + data.response.audio);
                            await audio.play();
                        } catch (audioError) {
                            console.error('Error playing audio:', audioError);
                        }
                    }
                } else {
                    addMessage('Error: Invalid response format', false);
                }
            } catch (error) {
                console.error('Error:', error);
                addMessage('Error: Could not send message. Please try again.', false);
            }
        }
        
        // Initial greeting
        addMessage('Hello! I am the Subliminal AI Agent. How can I assist you today?', false);
    </script>
</body>
</html>
""")

@app.route('/')
def home():
    """Render the chat interface."""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat interactions with neural modulation."""
    try:
        if not FEATURES['neural_modulation']:
            return jsonify({'error': 'Neural modulation is disabled'}), 403

        data = request.get_json()
        if not data:
            return jsonify({
                'response': {
                    'text': "I didn't receive your message clearly. Could you please try again?",
                    'audio': None
                },
                'metrics': agent.get_interaction_metrics()
            }), 200
            
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({
                'response': {
                    'text': "I didn't catch that. What would you like to discuss?",
                    'audio': None
                },
                'metrics': agent.get_interaction_metrics()
            }), 200

        # Generate agent response with neural patterns
        response = agent.generate_response(user_message, {})
        
        # Convert neural pattern to audio if enabled
        if FEATURES['audio_synthesis'] and 'neural_pattern' in response:
            try:
                audio_signal = response['neural_pattern']
                # Normalize and scale audio
                audio_signal = np.int16(audio_signal * 32767)
                
                # Convert to bytes for transmission
                import base64
                import io
                import wave
                
                byte_buffer = io.BytesIO()
                with wave.open(byte_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(NEURAL_CONFIG['sample_rate'])
                    wav_file.writeframes(audio_signal.tobytes())
                
                response['audio'] = base64.b64encode(byte_buffer.getvalue()).decode('utf-8')
            except Exception as audio_error:
                logger.error(f"Error generating audio: {str(audio_error)}")
                response['audio'] = None
        
        metrics = agent.get_interaction_metrics()
        
        return jsonify({
            'response': response,
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Error in chat interaction: {str(e)}")
        return jsonify({
            'response': {
                'text': "I'm experiencing a moment of reflection. Please share your thoughts again.",
                'audio': None
            },
            'metrics': agent.get_interaction_metrics()
        }), 200

@app.route('/api/generate/binaural', methods=['POST'])
def generate_binaural():
    """Generate binaural beats for brainwave entrainment."""
    if not FEATURES['neural_modulation']:
        return jsonify({'error': 'Neural modulation is disabled'}), 403

    data = request.get_json()
    base_freq = float(data.get('base_freq', 100))
    beat_freq = float(data.get('beat_freq', 10))
    
    try:
        left, right = oscillator.create_binaural_beat(base_freq, beat_freq)
        return jsonify({
            'left_channel': left.tolist(),
            'right_channel': right.tolist()
        })
    except Exception as e:
        logger.error(f"Error generating binaural beats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate/entrainment', methods=['POST'])
def generate_entrainment():
    """Generate entrainment sequence between brainwave states."""
    if not FEATURES['neural_modulation']:
        return jsonify({'error': 'Neural modulation is disabled'}), 403

    data = request.get_json()
    start_state = BrainwaveStates[data.get('start_state', 'BETA')]
    target_state = BrainwaveStates[data.get('target_state', 'ALPHA')]
    duration = int(data.get('duration', NEURAL_CONFIG['duration']))
    
    try:
        signal = oscillator.create_entrainment_sequence(
            start_state,
            target_state,
            duration
        )
        
        # Apply psychological modulation if enabled
        if data.get('apply_modulation', False):
            signal = oscillator.apply_psychological_modulation(signal)
            
        return jsonify({
            'signal': signal.tolist(),
            'sample_rate': NEURAL_CONFIG['sample_rate']
        })
    except Exception as e:
        logger.error(f"Error generating entrainment sequence: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate/rogue', methods=['POST'])
def generate_rogue_pattern():
    """Generate rogue pattern for controlled studies."""
    if not FEATURES['neural_modulation']:
        return jsonify({'error': 'Neural modulation is disabled'}), 403

    data = request.get_json()
    base_freq = float(data.get('base_freq', 10))
    influence_level = float(data.get('influence_level', 0.3))
    
    try:
        # Generate base signal
        base_signal = oscillator.generate_frequency(base_freq)
        
        # Apply rogue pattern
        rogue_signal = oscillator.generate_rogue_pattern(
            base_signal,
            influence_level
        )
        
        return jsonify({
            'signal': rogue_signal.tolist(),
            'sample_rate': NEURAL_CONFIG['sample_rate']
        })
    except Exception as e:
        logger.error(f"Error generating rogue pattern: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
