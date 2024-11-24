import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import keyboard
import pyautogui
import tempfile
import torch
import os
import pyperclip
import warnings
import pdb

REC_DEVICE = 1
CHANNELS = 1
SAMPLE_RATE = 44100
KEYS = ["ctrl", "f2"]
COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def keys_pressed():
    for i in KEYS:
        if not keyboard.is_pressed(i):
            return False
    return True


def infinite_rec(sample_rate=SAMPLE_RATE, channels=CHANNELS):
    audio_list = []

    def callback(indata, frames, time, status):
        audio_list.append(indata.copy())

    with sd.InputStream(device=REC_DEVICE, samplerate=sample_rate, blocksize=2048, channels=channels, callback=callback):
        print("Recording...")
        try:
            while keys_pressed():
                pass
            print("Done")
            return audio_list
        except Exception as e:
            print(f"Error: {e}")


def main():
    print("Loading whisper...")
    whisper_model = whisper.load_model("large")
    whisper_model.to(COMPUTE_DEVICE)
    print("Done")
    print(f"Press { str(' + '.join(KEYS)) } to transcribe")
    try:
        while True:
            if keys_pressed():
                temp_filename = tempfile.mktemp(suffix='.wav', dir='./output')
                audio_list = infinite_rec()
                audio_array = np.concatenate(audio_list)
                sf.write(temp_filename, audio_array, SAMPLE_RATE)
                full_path = os.path.abspath(temp_filename)
                print("Transcribing...")
                result = whisper_model.transcribe(full_path)
                pyperclip.copy(result["text"])
                pyautogui.hotkey("ctrl", "v")
                print("Done")
                print(f"Press { str(' + '.join(KEYS)) } to transcribe")
    except KeyboardInterrupt:
        print("Goodbye\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()