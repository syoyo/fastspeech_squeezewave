# We retain the copyright notice by NVIDIA from the original code. However, we
# we reserve our rights on the modifications based on the original code.
#
# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
#from scipy.io.wavfile import write
import torch
#from mel2samp import files_to_list, MAX_WAV_VALUE
#from denoiser import Denoiser
import time
import json

from glow import SqueezeWave, SqueezeWaveLoss


config_json= """
{
        "n_mel_channels": 80,
        "n_flows": 12,
        "n_audio_channel": 256,
        "n_early_every": 2,
        "n_early_size": 16,
        "WN_config": {
            "n_layers": 8,
            "n_channels": 128,
            "kernel_size": 3
        }
}
"""

squeezewave_config = json.loads(config_json)

model = SqueezeWave(**squeezewave_config)
print(model)

trace = torch.jit.script(model)

