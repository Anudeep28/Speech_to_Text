import pyaudio
import wave
import threading
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# Audio recording parameters
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000

class AudioRecorder:
    def __init__(self, output_directory):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.thread = None
        self.output_directory = output_directory

    def start_recording(self):
        self.frames = []
        self.is_recording = True
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.thread:
            self.thread.join()

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
            return False

        full_path = os.path.join(self.output_directory, filename)
        wf = wave.open(full_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        return True

    def __del__(self):
        self.audio.terminate()

class RecorderGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Recorder")
        self.master.geometry("300x200")

        self.output_directory = ""
        self.recorder = None

        self.filename_label = tk.Label(master, text="Filename:")
        self.filename_label.pack()

        self.filename_entry = tk.Entry(master)
        self.filename_entry.pack()

        self.start_button = tk.Button(master, text="Start Recording", command=self.start_recording)
        self.start_button.pack()

        self.stop_button = tk.Button(master, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack()

        self.quit_button = tk.Button(master, text="Quit", command=self.quit)
        self.quit_button.pack()

        self.select_dir_button = tk.Button(master, text="Select Output Directory", command=self.select_directory)
        self.select_dir_button.pack()

    def select_directory(self):
        self.output_directory = filedialog.askdirectory()
        if self.output_directory:
            self.recorder = AudioRecorder(self.output_directory)
            messagebox.showinfo("Directory Selected", f"Output directory set to: {self.output_directory}")

    def start_recording(self):
        if not self.recorder:
            messagebox.showerror("Error", "Please select an output directory first.")
            return

        filename = self.filename_entry.get()
        if not filename:
            messagebox.showerror("Error", "Please enter a filename.")
            return

        if not filename.endswith('.wav'):
            filename += '.wav'

        self.recorder.start_recording()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def stop_recording(self):
        self.recorder.stop_recording()
        filename = self.filename_entry.get()
        if not filename.endswith('.wav'):
            filename += '.wav'

        if self.recorder.save_audio(filename):
            messagebox.showinfo("Success", f"Recording saved as {filename}")
        else:
            messagebox.showerror("Error", "No audio data to save.")

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def quit(self):
        if self.recorder and self.recorder.is_recording:
            self.recorder.stop_recording()
        self.master.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderGUI(root)
    root.mainloop()
