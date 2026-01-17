import numpy as np
from scipy.io import wavfile

# Your data
frequencies = np.array([440, 880, 1320])  # Hz
amplitudes = np.array([0.5, 0.3, 0.2])    # Relative amplitudes

# Audio parameters
sample_rate = 44100  # CD quality
duration = 2.0       # seconds

# Create time array
t = np.linspace(0, duration, int(sample_rate * duration))

# Generate composite waveform by summing all frequency components
composite_wave = np.zeros_like(t)
for freq, amp in zip(frequencies, amplitudes):
    composite_wave += amp * np.sin(2 * np.pi * freq * t)

# Normalize to prevent clipping
composite_wave = composite_wave / np.max(np.abs(composite_wave))

# Convert to 16-bit PCM
audio_data = np.int16(composite_wave * 32767)

# Save to file
wavfile.write('composite_sound.wav', sample_rate, audio_data)