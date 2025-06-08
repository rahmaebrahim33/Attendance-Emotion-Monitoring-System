import numpy as np
import pygame
import os
from scipy.io import wavfile

def generate_tone(freq, duration, sample_rate=22050):
    """Generate a simple tone at the given frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(freq * 2 * np.pi * t)
    # Add fade in/out
    fade = 0.05  # 50ms fade
    fade_samples = int(fade * sample_rate)
    tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return (tone * 32767).astype(np.int16)

def create_attendance_sound():
    """Create a sound for marking attendance - two ascending beeps."""
    sample_rate = 22050
    tone1 = generate_tone(440, 0.1, sample_rate)  # A4
    tone2 = generate_tone(880, 0.2, sample_rate)  # A5
    silence = np.zeros(int(0.05 * sample_rate), dtype=np.int16)
    sound = np.concatenate([tone1, silence, tone2])
    return sound, sample_rate

def create_face_detected_sound():
    """Create a sound for face detection - single short beep."""
    sample_rate = 22050
    sound = generate_tone(700, 0.1, sample_rate)
    return sound, sample_rate

def create_unknown_person_sound():
    """Create a sound for unknown person - two descending beeps."""
    sample_rate = 22050
    tone1 = generate_tone(880, 0.1, sample_rate)  # A5
    tone2 = generate_tone(440, 0.2, sample_rate)  # A4
    silence = np.zeros(int(0.05 * sample_rate), dtype=np.int16)
    sound = np.concatenate([tone1, silence, tone2])
    return sound, sample_rate

def create_calibration_sound():
    """Create a sound for calibration complete - three ascending tones."""
    sample_rate = 22050
    tone1 = generate_tone(440, 0.1, sample_rate)  # A4
    tone2 = generate_tone(550, 0.1, sample_rate)  # C#5
    tone3 = generate_tone(660, 0.2, sample_rate)  # E5
    silence = np.zeros(int(0.05 * sample_rate), dtype=np.int16)
    sound = np.concatenate([tone1, silence, tone2, silence, tone3])
    return sound, sample_rate

def main():
    """Generate and save all sound files."""
    sound_dir = "sounds"
    if not os.path.exists(sound_dir):
        os.makedirs(sound_dir)
        print(f"Created directory: {sound_dir}")
    
    # Generate sound files
    sound_generators = {
        "attendance_marked.wav": create_attendance_sound,
        "face_detected.wav": create_face_detected_sound,
        "unknown_person.wav": create_unknown_person_sound,
        "calibration_complete.wav": create_calibration_sound
    }
    
    for filename, generator in sound_generators.items():
        sound, sample_rate = generator()
        filepath = os.path.join(sound_dir, filename)
        wavfile.write(filepath, sample_rate, sound)
        print(f"Created sound file: {filepath}")

if __name__ == "__main__":
    main() 