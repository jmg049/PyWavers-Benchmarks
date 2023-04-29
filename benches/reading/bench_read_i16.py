import pyperf
import os


def files_from_base(base):
    data = []
    for currentpath, _, files in os.walk(base):
        for file in files:
            data.append(os.path.join(currentpath, file))
    return data


def benchmark_i16_8000(runner):
    BASE = './resources/quickstart_genspeech_i16_8000/LPCNet_listening_test'
    DATA = files_from_base(BASE)
    runner.timeit(
        "ScipyIO-i16",
        stmt='for f in DATA: scipy.io.wavfile.read(f)[1].astype(np.int16)',
        setup=f"import scipy;import numpy as np; DATA = {DATA}")

    runner.timeit(
        "TorchAudio-i16-Sox",
        stmt='for f in DATA: torchaudio.load(f)[0].type(torch.ShortTensor)',
        setup=
        f"import torchaudio;torchaudio.set_audio_backend('sox_io');import torch; DATA = {DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Soundfile",
        stmt='for f in DATA: torchaudio.load(f)[0].type(torch.ShortTensor)',
        setup=
        f"import torchaudio;torchaudio.set_audio_backend('soundfile');import torch; DATA = {DATA}"
    )

    runner.timeit("PyWavers-i16",
                  stmt='for f in DATA: pywavers.read(f, dtype=np.int16)',
                  setup=f"import pywavers; import numpy as np; DATA = {DATA}")
    runner.timeit("Soundfile-i16",
                  stmt='for f in DATA: soundfile.read(f, dtype=np.int16)',
                  setup=f"import soundfile;import numpy as np; DATA = {DATA}")


def benchmark_i16_16000(runner):
    BASE = './resources/quickstart_genspeech_i16_16000/LPCNet_listening_test'
    DATA = files_from_base(BASE)
    runner.timeit(
        "ScipyIO-i16",
        stmt='for f in DATA: scipy.io.wavfile.read(f)[1].astype(np.int16)',
        setup=f"import scipy;import numpy as np; DATA = {DATA}")

    runner.timeit(
        "TorchAudio-i16-Sox",
        stmt='for f in DATA: torchaudio.load(f)[0].type(torch.ShortTensor)',
        setup=
        f"import torchaudio;torchaudio.set_audio_backend('sox_io');import torch; DATA = {DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Soundfile",
        stmt='for f in DATA: torchaudio.load(f)[0].type(torch.ShortTensor)',
        setup=
        f"import torchaudio;torchaudio.set_audio_backend('soundfile');import torch; DATA = {DATA}"
    )

    runner.timeit("PyWavers-i16",
                  stmt='for f in DATA: pywavers.read(f, dtype=np.int16)',
                  setup=f"import pywavers; import numpy as np; DATA = {DATA}")
    runner.timeit("Soundfile-i16",
                  stmt='for f in DATA: soundfile.read(f, dtype=np.int16)',
                  setup=f"import soundfile;import numpy as np; DATA = {DATA}")


def benchmark_i16_22050(runner):
    BASE = './resources/quickstart_genspeech_i16_22050/LPCNet_listening_test'
    DATA = files_from_base(BASE)
    runner.timeit(
        "ScipyIO-i16",
        stmt='for f in DATA: scipy.io.wavfile.read(f)[1].astype(np.int16)',
        setup=f"import scipy;import numpy as np; DATA = {DATA}")

    runner.timeit(
        "TorchAudio-i16-Sox",
        stmt='for f in DATA: torchaudio.load(f)[0].type(torch.ShortTensor)',
        setup=
        f"import torchaudio;torchaudio.set_audio_backend('sox_io');import torch; DATA = {DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Soundfile",
        stmt='for f in DATA: torchaudio.load(f)[0].type(torch.ShortTensor)',
        setup=
        f"import torchaudio;torchaudio.set_audio_backend('soundfile');import torch; DATA = {DATA}"
    )

    runner.timeit("PyWavers-i16",
                  stmt='for f in DATA: pywavers.read(f, dtype=np.int16)',
                  setup=f"import pywavers; import numpy as np; DATA = {DATA}")
    runner.timeit("Soundfile-i16",
                  stmt='for f in DATA: soundfile.read(f, dtype=np.int16)',
                  setup=f"import soundfile;import numpy as np; DATA = {DATA}")


def benchmark_i16_44100(runner):
    BASE = './resources/quickstart_genspeech_i16_44100/LPCNet_listening_test'
    DATA = files_from_base(BASE)
    runner.timeit(
        "ScipyIO-i16",
        stmt='for f in DATA: scipy.io.wavfile.read(f)[1].astype(np.int16)',
        setup=f"import scipy;import numpy as np; DATA = {DATA}")

    runner.timeit(
        "TorchAudio-i16-Sox",
        stmt='for f in DATA: torchaudio.load(f)[0].type(torch.ShortTensor)',
        setup=
        f"import torchaudio;torchaudio.set_audio_backend('sox_io');import torch; DATA = {DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Soundfile",
        stmt='for f in DATA: torchaudio.load(f)[0].type(torch.ShortTensor)',
        setup=
        f"import torchaudio;torchaudio.set_audio_backend('soundfile');import torch; DATA = {DATA}"
    )

    runner.timeit("PyWavers-i16",
                  stmt='for f in DATA: pywavers.read(f, dtype=np.int16)',
                  setup=f"import pywavers; import numpy as np; DATA = {DATA}")
    runner.timeit("Soundfile-i16",
                  stmt='for f in DATA: soundfile.read(f, dtype=np.int16)',
                  setup=f"import soundfile;import numpy as np; DATA = {DATA}")


if __name__ == '__main__':
    runner = pyperf.Runner(values=20, processes=8, loops=10)

    benchmark_i16_8000(runner)
    benchmark_i16_16000(runner)
    benchmark_i16_22050(runner)
    benchmark_i16_44100(runner)