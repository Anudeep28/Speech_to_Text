import pyaudio
import wave
import threading
import time
import os

# Audio recording parameters
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.thread = None

    def start_recording(self):
        self.frames = []
        self.is_recording = True
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
        print("Recording started...")

    def stop_recording(self):
        self.is_recording = False
        if self.thread:
            self.thread.join()
        print("Recording stopped.")

    def _record(self):
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )

        while self.is_recording:
            data = self.stream.read(FRAMES_PER_BUFFER)
            self.frames.append(data)

        self.stream.stop_stream()
        self.stream.close()

    def save_audio(self, filename):
        if not self.frames:
            print("No audio data to save.")
            return
        #print("path:",os.getcwd())
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"Audio saved as {filename}")

    def __del__(self):
        self.audio.terminate()

def main():
    recorder = AudioRecorder()
    recording_number = 1

    while True:
        command = input("Enter 'start' to begin recording, 'stop' to end recording, or 'quit' to exit: ").lower()

        if command == 'start':
            recorder.start_recording()
        elif command == 'stop':
            recorder.stop_recording()
            filename = f"dataset_anu/audio/file{recording_number}.wav"
            recorder.save_audio(filename)
            recording_number += 1
        elif command == 'quit':
            break
        else:
            print("Invalid command. Please try again.")

    print("Program ended.")

if __name__ == "__main__":
    main()
