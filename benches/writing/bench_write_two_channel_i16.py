import subprocess
import pyperf
import os


def files_from_base(base):
    data = []
    for currentpath, _, files in os.walk(base):
        for file in files:
            data.append(os.path.join(currentpath, file))
    return data


def benchmark_i16_two_channels_8000(runner):
    BASE = './resources/quickstart_genspeech_i16_two_channel_8000/LPCNet_listening_test'
    DATA = files_from_base(BASE)
    OUT_DATA = [f.replace('./resources', './temp_files') for f in DATA]
    os.makedirs(BASE.replace('./resources', './temp_files'), exist_ok=True)

    runner.timeit(
        "ScipyIO-i16-8000",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data = scipy.io.wavfile.read(f)[1].astype(np.int16);os.makedirs(Path(o).parents[0], exist_ok=True);scipy.io.wavfile.write(o, 8000, data)',
        setup=
        f"import os; from pathlib import Path;import scipy;import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Sox-8000",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=torchaudio.load(f)[0].type(torch.ShortTensor);os.makedirs(Path(o).parents[0], exist_ok=True);torchaudio.save(o,  data, 8000, encoding="PCM_S", bits_per_sample=16)',
        setup=
        f"import os; from pathlib import Path;import torchaudio;torchaudio.set_audio_backend('sox_io');import torch; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Soundfile-8000",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=torchaudio.load(f)[0].type(torch.ShortTensor);os.makedirs(Path(o).parents[0], exist_ok=True);torchaudio.save(o, data, 8000, encoding="PCM_S", bits_per_sample=16)',
        setup=
        f"import os; from pathlib import Path;import torchaudio;torchaudio.set_audio_backend('soundfile');import torch; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )

    ## TODO: Investigate this, trying to load all files sequentially causes a seg fault while reading. However, this does not happend when reading the files individually.
    # runner.timeit(
    #     "PyWavers-i16-8000",
    #     stmt=
    #     'for f, o in zip(DATA, OUT_DATA): data= pywavers.read(f, dtype=np.int16);pywavers.write(o, data, 8000, np.int16)',
    #     setup=
    #     f"import os; from pathlib import Path;import pywavers; import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    # )
    runner.timeit(
        "Soundfile-i16-8000",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=soundfile.read(f, dtype=np.int16)[0];os.makedirs(Path(o).parents[0], exist_ok=True);soundfile.write(o, data, 8000, subtype="PCM_16")',
        setup=
        f"import os; from pathlib import Path;import soundfile;import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )


def benchmark_i16_two_channels_16000(runner):
    BASE = './resources/quickstart_genspeech_i16_two_channel_16000/LPCNet_listening_test'

    DATA = files_from_base(BASE)
    OUT_DATA = [f.replace('./resources', './temp_files') for f in DATA]

    os.makedirs(BASE.replace('./resources', './temp_files'), exist_ok=True)

    runner.timeit(
        "ScipyIO-i16-16000",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data = scipy.io.wavfile.read(f)[1].astype(np.int16);os.makedirs(Path(o).parents[0], exist_ok=True);scipy.io.wavfile.write(o, 16000, data)',
        setup=
        f"import os; from pathlib import Path;import scipy;import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Sox-16000",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=torchaudio.load(f)[0].type(torch.ShortTensor);os.makedirs(Path(o).parents[0], exist_ok=True);torchaudio.save(o,  data, 16000, encoding="PCM_S", bits_per_sample=16)',
        setup=
        f"import os; from pathlib import Path;import torchaudio;torchaudio.set_audio_backend('sox_io');import torch; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Soundfile-16000",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=torchaudio.load(f)[0].type(torch.ShortTensor);os.makedirs(Path(o).parents[0], exist_ok=True);torchaudio.save(o, data, 16000, encoding="PCM_S", bits_per_sample=16)',
        setup=
        f"import os; from pathlib import Path;import torchaudio;torchaudio.set_audio_backend('soundfile');import torch; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )

    ## TODO: Investigate this, trying to load all files sequentially causes a seg fault while reading. However, this does not happend when reading the files individually.
    # runner.timeit(
    #     "PyWavers-i16-16000",
    #     stmt=
    #     'for f, o in zip(DATA, OUT_DATA): data= pywavers.read(f, dtype=np.int16);os.makedirs(Path(o).parents[0], exist_ok=True);pywavers.write(o, data, 16000, np.int16)',
    #     setup=
    #     f"import os; from pathlib import Path;import pywavers; import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    # )
    runner.timeit(
        "Soundfile-i16",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=soundfile.read(f, dtype=np.int16)[0];os.makedirs(Path(o).parents[0], exist_ok=True);soundfile.write(o, data, 16000, subtype="PCM_16")',
        setup=
        f"import os; from pathlib import Path;import soundfile;import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )



def benchmark_i16_two_channels_22050(runner):
    BASE = './resources/quickstart_genspeech_i16_two_channel_22050/LPCNet_listening_test'
    DATA = files_from_base(BASE)
    OUT_DATA = [f.replace('./resources', './temp_files') for f in DATA]
    os.makedirs(BASE.replace('./resources', './temp_files'), exist_ok=True)

    runner.timeit(
        "ScipyIO-i16-22050",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data = scipy.io.wavfile.read(f)[1].astype(np.int16);os.makedirs(Path(o).parents[0], exist_ok=True);scipy.io.wavfile.write(o, 22050, data)',
        setup=
        f"import os; from pathlib import Path;import scipy;import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Sox-22050",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=torchaudio.load(f)[0].type(torch.ShortTensor);os.makedirs(Path(o).parents[0], exist_ok=True);torchaudio.save(o,  data, 22050, encoding="PCM_S", bits_per_sample=16)',
        setup=
        f"import os; from pathlib import Path;import torchaudio;torchaudio.set_audio_backend('sox_io');import torch; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Soundfile-22050",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=torchaudio.load(f)[0].type(torch.ShortTensor);os.makedirs(Path(o).parents[0], exist_ok=True);torchaudio.save(o, data, 22050, encoding="PCM_S", bits_per_sample=16)',
        setup=
        f"import os; from pathlib import Path;import torchaudio;torchaudio.set_audio_backend('soundfile');import torch; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )

    # runner.timeit(
    #     "PyWavers-i16-22050",
    #     stmt=
    #     'for f, o in zip(DATA, OUT_DATA): data= pywavers.read(f, dtype=np.int16);os.makedirs(Path(o).parents[0], exist_ok=True);pywavers.write(o, data, 22050, np.int16)',
    #     setup=
    #     f"import os; from pathlib import Path;import pywavers; import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    # )
    runner.timeit(
        "Soundfile-i16-22050",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=soundfile.read(f, dtype=np.int16)[0];os.makedirs(Path(o).parents[0], exist_ok=True);soundfile.write(o, data, 22050, subtype="PCM_16")',
        setup=
        f"import os; from pathlib import Path;import soundfile;import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )



def benchmark_i16_two_channels_44100(runner):
    BASE = './resources/quickstart_genspeech_i16_two_channel_44100/LPCNet_listening_test'
    DATA = files_from_base(BASE)
    OUT_DATA = [f.replace('./resources', './temp_files') for f in DATA]

    OUT_BASE = BASE.replace('./resources', './temp_files')

    os.makedirs(BASE.replace('./resources', './temp_files'), exist_ok=True)

    runner.timeit(
        "ScipyIO-i16-44100",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data = scipy.io.wavfile.read(f)[1].astype(np.int16);os.makedirs(Path(o).parents[0], exist_ok=True);scipy.io.wavfile.write(o, 44100, data)',
        setup=
        f"import os; from pathlib import Path;import scipy;import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Sox-44100",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=torchaudio.load(f)[0].type(torch.ShortTensor);os.makedirs(Path(o).parents[0], exist_ok=True);torchaudio.save(o,  data, 44100, encoding="PCM_S", bits_per_sample=16)',
        setup=
        f"import os; from pathlib import Path;import torchaudio;torchaudio.set_audio_backend('sox_io');import torch; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )
    runner.timeit(
        "TorchAudio-i16-Soundfile-44100",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=torchaudio.load(f)[0].type(torch.ShortTensor);os.makedirs(Path(o).parents[0], exist_ok=True);torchaudio.save(o, data, 44100, encoding="PCM_S", bits_per_sample=16)',
        setup=
        f"import os; from pathlib import Path;import torchaudio;torchaudio.set_audio_backend('soundfile');import torch; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )

    # runner.timeit(
    #     "PyWavers-i16-44100",
    #     stmt=
    #     'for f, o in zip(DATA, OUT_DATA): data= pywavers.read(f, dtype=np.int16);os.makedirs(Path(o).parents[0], exist_ok=True);pywavers.write(o, data, 44100, np.int16)',
    #     setup=
    #     f"import os; from pathlib import Path;import pywavers; import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    # )
    runner.timeit(
        "Soundfile-i16-44100",
        stmt=
        'for f, o in zip(DATA, OUT_DATA): data=soundfile.read(f, dtype=np.int16)[0];os.makedirs(Path(o).parents[0], exist_ok=True);soundfile.write(o, data, 44100, subtype="PCM_16")',
        setup=
        f"import os; from pathlib import Path;import soundfile;import numpy as np; DATA = {DATA};OUT_DATA={OUT_DATA}"
    )


if __name__ == '__main__':
    runner = pyperf.Runner(values=20, processes=8, loops=10)

    benchmark_i16_two_channels_8000(runner)
    benchmark_i16_two_channels_16000(runner)
    benchmark_i16_two_channels_22050(runner)
    benchmark_i16_two_channels_44100(runner)
    # run cleanup script
    subprocess.run(['sh', 'cleanup.sh'])