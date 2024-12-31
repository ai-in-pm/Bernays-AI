from enum import Enum
import numpy as np

class BrainwaveStates(Enum):
    DELTA = "DELTA"      # 0.5-4 Hz: Deep sleep
    THETA = "THETA"      # 4-8 Hz: Meditation, memory
    ALPHA = "ALPHA"      # 8-13 Hz: Relaxation
    BETA = "BETA"        # 13-30 Hz: Active thinking
    GAMMA = "GAMMA"      # 30-100 Hz: Peak concentration

class NeuralOscillator:
    def __init__(self, config):
        """Initialize the neural oscillator with configuration."""
        self.sample_rate = config['sample_rate']
        self.min_freq = config['min_freq']
        self.max_freq = config['max_freq']
        self.duration = config['duration']
        
    def generate_frequency(self, frequency, duration=None):
        """Generate a simple sine wave at the specified frequency."""
        if duration is None:
            duration = self.duration
            
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        return np.sin(2 * np.pi * frequency * t)
        
    def create_binaural_beat(self, base_freq, beat_freq):
        """Create a binaural beat with specified base and beat frequencies."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        left_channel = np.sin(2 * np.pi * base_freq * t)
        right_channel = np.sin(2 * np.pi * (base_freq + beat_freq) * t)
        return left_channel, right_channel
        
    def create_isochronic_tone(self, frequency):
        """Create an isochronic tone at specified frequency."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        carrier = np.sin(2 * np.pi * frequency * 10 * t)  # Higher frequency carrier
        modulator = 0.5 * (1 + np.sin(2 * np.pi * frequency * t))  # Amplitude modulation
        return carrier * modulator
        
    def create_entrainment_sequence(self, start_state, target_state):
        """Create a frequency entrainment sequence between brainwave states."""
        frequencies = {
            BrainwaveStates.DELTA: 2,
            BrainwaveStates.THETA: 6,
            BrainwaveStates.ALPHA: 10,
            BrainwaveStates.BETA: 20,
            BrainwaveStates.GAMMA: 40
        }
        
        start_freq = frequencies[start_state]
        target_freq = frequencies[target_state]
        
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        freq_ramp = np.linspace(start_freq, target_freq, len(t))
        phase = 2 * np.pi * np.cumsum(freq_ramp) / self.sample_rate
        return np.sin(phase)
