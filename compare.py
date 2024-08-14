import torch
import wave
import numpy as np
from vad.collections.silero import SileroNCNN

torch.set_grad_enabled(False)

old_model = 'vad/collections/resources/silero_vad.jit' # https://github.com/snakers4/silero-vad/blob/v4.0stable/files/silero_vad.jit
model = torch.jit.load(old_model, map_location='cpu')
old_jit_model = model._model # only 16k

new_model = 'vad/collections/resources/silero.jit'
new_jit_model = torch.jit.load(new_model, map_location='cpu')

ncnn_model = SileroNCNN()

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

h_old = torch.zeros(2, 1, 64)
c_old = torch.zeros(2, 1, 64)
h_new = torch.zeros(2, 1, 64)
c_new = torch.zeros(2, 1, 64)

chunk_size = int(0.032 * wave_file_sample_rate)  # 0.032 seconds
start = 0
while start < samples_float32.shape[0]:
    end = start + chunk_size
    end = min(end, samples_float32.shape[0])
    x = torch.from_numpy(samples_float32[start:end]).reshape(1, -1)

    y_old, h_old, c_old = old_jit_model(x, h_old, c_old)
    y_new, h_new, c_new = new_jit_model(x, h_new, c_new)
    y_ncnn = ncnn_model.inference(samples_float32[start:end])
    start = end

    print(f'old:{y_old}, new:{y_new}, ncnn:{y_ncnn}')