import os
from scipy.io.wavfile import write
import torch
from mel2samp import files_to_list, MAX_WAV_VALUE
#from denoiser import Denoiser
import time
import json

from glow import SqueezeWave, SqueezeWaveLoss

device = torch.device('cpu')
mel_file = "mel_spectrograms/test.pt"

# a128-c128
config_json= """
{
        "n_mel_channels": 80,
        "n_flows": 12,
        "n_audio_channel": 128,
        "n_early_every": 2,
        "n_early_size": 16,
        "WN_config": {
            "n_layers": 8,
            "n_channels": 128,
            "kernel_size": 3
        }
}
"""

#model = torch.load("pretrain_models/L128_small_pretrain", map_location=device)['model']
#print(model.state_dict())
#torch.save(model.state_dict(), "weights.pt")

squeezewave_config = json.loads(config_json)

model = SqueezeWave(**squeezewave_config)
model.load_state_dict(torch.load("weights.pt", map_location=device))
#print(model.state_dict())
print("OK")

#trace = torch.jit.script(model)
#trace.save("squeezewave.script.pt")

mel = torch.load(mel_file,map_location=device)
#mel = torch.autograd.Variable(mel)
mel = torch.tensor(mel)
mel = mel.half() 
print(mel)

sigma = 0.6
sampling_rate = 22050
output_dir="."

with torch.no_grad():
    audio = model.forward(mel, sigma=sigma)
    print(audio)

    audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    audio_path = os.path.join(
        output_dir, "bora_synthesis.wav")
    write(audio_path, sampling_rate, audio)

#trace = torch.jit.trace(model, mel)

