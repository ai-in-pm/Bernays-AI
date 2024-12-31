import numpy as np
from scipy import signal
import sounddevice as sd
from typing import Dict, Tuple, Optional
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class BrainwaveStates(Enum):
    DELTA = (0.5, 4)    # Deep sleep
    THETA = (4, 8)      # Deep relaxation, meditation
    ALPHA = (8, 13)     # Relaxed awareness
    BETA = (13, 30)     # Active thinking
    GAMMA = (30, 40)    # High-level cognition

class NeuralOscillator:
    def __init__(self, config: Dict):
        """
        Initialize the neural oscillator with configuration parameters.
        
        Args:
            config (Dict): Configuration dictionary containing:
                - sample_rate: Sampling rate in Hz
                - min_freq: Minimum frequency in Hz
                - max_freq: Maximum frequency in Hz
                - duration: Duration in seconds
        """
        self.sample_rate = config['sample_rate']
        self.min_freq = config['min_freq']
        self.max_freq = config['max_freq']
        self.duration = config['duration']
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        logger.info(f"Initialized NeuralOscillator with config: {config}")

    def generate_frequency(self, freq: float, amplitude: float = 1.0) -> np.ndarray:
        """Generate a pure sine wave at specified frequency using scipy."""
        return amplitude * signal.chirp(self.t, f0=freq, f1=freq, t1=self.duration, method='linear')

    def create_binaural_beat(self, base_freq: float, beat_freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create binaural beat by generating two slightly different frequencies.
        
        Args:
            base_freq (float): Base frequency for one ear
            beat_freq (float): Desired beat frequency (difference between ears)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Left and right ear signals
        """
        left_signal = self.generate_frequency(base_freq)
        right_signal = self.generate_frequency(base_freq + beat_freq)
        return left_signal, right_signal

    def create_isochronic_tone(self, freq: float, duty_cycle: float = 0.5) -> np.ndarray:
        """
        Create isochronic tone using scipy's square wave modulation.
        
        Args:
            freq (float): Base frequency
            duty_cycle (float): Proportion of time the tone is on (0-1)
            
        Returns:
            np.ndarray: Pulsed signal
        """
        carrier = self.generate_frequency(freq)
        pulse_freq = freq / 10  # Slower pulsing rate
        pulse = signal.square(2 * np.pi * pulse_freq * self.t, duty=duty_cycle)
        return carrier * ((pulse + 1) / 2)  # Normalize pulse to 0-1 range

    def create_entrainment_sequence(self, 
                                  start_state: BrainwaveStates,
                                  target_state: BrainwaveStates,
                                  transition_duration: Optional[int] = None) -> np.ndarray:
        """
        Create a gradual entrainment sequence between brainwave states using scipy.
        
        Args:
            start_state (BrainwaveStates): Starting brainwave state
            target_state (BrainwaveStates): Target brainwave state
            transition_duration (Optional[int]): Duration of transition in seconds
            
        Returns:
            np.ndarray: Combined signal for entrainment
        """
        if transition_duration is None:
            transition_duration = self.duration

        start_freq = np.mean(start_state.value)
        target_freq = np.mean(target_state.value)
        
        # Use scipy's chirp to create smooth frequency transition
        return signal.chirp(
            self.t,
            f0=start_freq,
            f1=target_freq,
            t1=transition_duration,
            method='logarithmic'
        )

    def apply_psychological_modulation(self, signal_input: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        Apply psychological modulation using scipy's signal processing.
        
        Args:
            signal_input (np.ndarray): Input signal
            intensity (float): Modulation intensity (0-1)
            
        Returns:
            np.ndarray: Modulated signal
        """
        # Create amplitude modulation envelope
        envelope = 1 + intensity * signal.gaussian(len(self.t), std=len(self.t)/6)
        
        # Add harmonic overtones using scipy's signal processing
        harmonics = np.zeros_like(signal_input)
        for i in range(2, 5):
            harmonic = np.roll(signal_input, i)
            harmonic = signal.filtfilt([1.0/i], [1.0], harmonic)
            harmonics += harmonic
        
        modulated_signal = signal_input * envelope + intensity * harmonics
        return modulated_signal / np.max(np.abs(modulated_signal))

    def generate_rogue_pattern(self, base_signal: np.ndarray, influence_level: float = 0.3) -> np.ndarray:
        """
        Generate a rogue pattern using scipy's advanced signal processing.
        WARNING: This should only be used in controlled research environments.
        
        Args:
            base_signal (np.ndarray): Base signal to modify
            influence_level (float): Level of rogue influence (0-1)
            
        Returns:
            np.ndarray: Modified signal with rogue patterns
        """
        if influence_level > 0.7:
            logger.warning("High influence level detected in rogue pattern generation")
            
        # Create subliminal modulation using filtered noise
        noise = np.random.normal(0, 1, len(base_signal))
        subliminal = signal.filtfilt([0.01], [1.0], noise)
        
        # Generate pattern interrupts using scipy's wavelets
        pattern_interrupts = signal.ricker(len(base_signal), influence_level * 10)
        
        # Combine components with advanced filtering
        rogue_signal = (base_signal * (1 - influence_level) + 
                       subliminal * influence_level * 0.5 +
                       pattern_interrupts * influence_level * 0.5)
        
        # Apply final bandpass filter to ensure frequency constraints
        sos = signal.butter(10, [self.min_freq, self.max_freq], 
                          btype='band', fs=self.sample_rate, output='sos')
        filtered_signal = signal.sosfilt(sos, rogue_signal)
        
        return filtered_signal / np.max(np.abs(filtered_signal))

    def play_audio(self, audio_signal: np.ndarray, duration: Optional[float] = None) -> None:
        """
        Play the generated audio signal through sounddevice.
        
        Args:
            audio_signal (np.ndarray): Audio signal to play
            duration (Optional[float]): Duration in seconds
        """
        if duration is None:
            duration = self.duration
            
        try:
            sd.play(audio_signal, self.sample_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")
            raise
