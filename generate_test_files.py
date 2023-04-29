import soundfile as sf
import numpy as np

import os
from tqdm import tqdm


sample_rates = [8000, 16000, 22050, 44100]
types = [np.int16, np.float32]
type_names = {
    np.int16: "i16",
    np.float32: "f32"
}

def get_all_files(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files



def generate_test_files(input_dir, output_dir_prefix):
    orig_files = get_all_files(input_dir)
    print(orig_files)
    for orig_file in tqdm(orig_files, desc="Processing files"):
        for sample_rate in sample_rates:
            for t in types:
                data, sr = sf.read(orig_file, dtype=t)
                out_file_name = orig_file.replace(input_dir, '')
                os.makedirs(os.path.dirname(f"{output_dir_prefix}_{type_names[t]}_{sample_rate}{out_file_name}"), exist_ok=True)
                sf.write(f"{output_dir_prefix}_{type_names[t]}_{sample_rate}{out_file_name}", data, sample_rate, subtype="PCM_16" if t == np.int16 else "FLOAT")

if __name__ == "__main__":
    generate_test_files("./resources/quickstart_genspeech", "./resources/quickstart_genspeech")
