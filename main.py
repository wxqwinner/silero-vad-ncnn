import wave
import numpy as np
from vad.collections.silero import SileroNCNN

model = SileroNCNN()

filename = "./en_example.wav"
with wave.open(filename) as f:
    wave_file_sample_rate = f.getframerate()
    num_channels = f.getnchannels()
    assert f.getsampwidth() == 2, f.getsampwidth()
    num_samples = f.getnframes()
    samples = f.readframes(num_samples)
    samples_int16 = np.frombuffer(samples, dtype=np.int16)
    samples_int16 = samples_int16.reshape(-1, num_channels)[:, 0]
    samples_float32 = samples_int16.astype(np.float32)
    samples_float32 = samples_float32 / 32768

chunk_size = int(0.032 * wave_file_sample_rate)  # 0.032 seconds
start = 0
while start < samples_float32.shape[0]:
    end = start + chunk_size
    end = min(end, samples_float32.shape[0])
    y = model.inference(samples_float32[start:end])
    start = end
    if y > 0.5:
        print('voice')
    else:
        print('unvoice')