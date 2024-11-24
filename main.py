import sounddevice as sd
import numpy as np
import whisper
import io
import keyboard
import argparse
import pyautogui

def main(pressed_key):
	print("test")
	model = whisper.load_model("tiny")
	samplerate = 44100
	duration = 1
	i = 0
	# Rec
	print("test")
	while i < 10:
		new_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
		sd.wait()
		audio_data = new_data.astype(np.float32)
		result = model.transcribe(audio_data)
		pyautogui.write(result)
		print(result)
		print("test")
		i += 1
	# Translate
	audio_data = audio_data.astype(np.float32)
	result = model.transcribe(audio_data)
	# Write
	pyautogui.write(result)

if __name__ == "__main__":
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--param1', type=str, help="Pressed key", required=True)
	# args = parser.parse_args()
	# main(args.param1)

	print("test")
	main("enter")

	print("test")

