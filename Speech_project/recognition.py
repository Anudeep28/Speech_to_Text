#################### Description ###############
"""
`Threading`: The Thread class from the threading module is imported to enable concurrent execution of code, allowing the program to run speech recognition in the background.
`Sleep`: The sleep function from the time module is used to pause the execution for a specified duration, which helps in managing the timing of recognition.
`Speech Recognition`: The speech_recognition library (aliased as sr) is imported to handle audio input and convert speech to text.
`Config`: A custom config module is imported, which presumably contains configuration values like energy thresholds and pause thresholds for the speech recognition process.
"""

from threading import Thread
from time import sleep
import speech_recognition as sr
import config


class SpeechRecognizer(object):

    STOP_PHRASES = [
        "stop recording",
        "stop listening",
    ]
    """
    `STOP_PHRASES`: A list of phrases that, when recognized, will stop the speech recognition process. This allows the user to control the program with voice commands.
    thread_count: A class variable to keep track of the number of threads currently running for speech recognition.
    """

    thread_count = 0

    def __init__(self, whisper_model: str, output_file_path: str) -> None:
        self.output_file_path = output_file_path
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self._recognized_speech = ""
        self._recognition_finished = False
        self._stop_listening = False
        self.full_recognized_speech = ""
        self.whisper_model = whisper_model
        """
        `output_file_path`: Path where the recognized speech will be saved.
        `recognizer`: An instance of the Recognizer class from the speech_recognition library to handle speech recognition tasks.
        `microphone`: An instance of the Microphone class to capture audio input.
        `_recognized_speech`: A string to store the recognized speech.
        `_recognition_finished`: A boolean flag to indicate whether the recognition process is complete.
        `_stop_listening`: A boolean flag to control the listening loop.
        `full_recognized_speech`: A string to accumulate all recognized speech.
        `whisper_model`: The model used for speech recognition, passed as an argument.

        """

        # Adjust microphone to the environment
        """
        This block adjusts the microphone settings for ambient noise, ensuring that the speech recognition
         is more accurate by setting various thresholds based on the configuration.

        """
        with self.microphone as source:
            print("Adjusting the microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            self.recognizer.dynamic_energy_threshold = False
            self.recognizer.energy_adjustment_during_recognition = False
            self.recognizer.energy_threshold = config.ENERGY_TRESHOLD
            self.recognizer.pause_threshold = config.PAUSE_TRESHOLD
            self.recognizer.phrase_threshold = config.PHRASE_TRESHOLD
            print("Microphone adjusted.")

    """
    This private method performs the actual speech recognition using the provided audio data.
    """
    def _recognize_speech(self, audio: sr.AudioData) -> None:
        """Used for the main speech recognition"""

        """
        The method attempts to recognize speech from the audio data using the specified whisper model. 
        If successful, it updates the recognized speech and sets the recognition status. If an error occurs, it simply returns None.
        """

        try:
            self._recognized_speech = self.recognizer.recognize_whisper(
                audio_data=audio, model=self.whisper_model
            )
            SpeechRecognizer.thread_count -= 1
            self._recognition_finished = True
            return
        except Exception:
            return None

    """
    This method allows speech recognition to run in a separate thread.
    """
    def _recognize_speech_in_background(self, audio: sr.AudioData) -> str:
        """Uses `self._recognize_speech` to recognize speech from the given audio"""

        # Start speech recognition
        thread = Thread(target=self._recognize_speech, args=(audio,))
        thread.start()
        SpeechRecognizer.thread_count += 1

        """
        A new thread is created to run the _recognize_speech method, allowing the main program
         to continue executing while waiting for recognition to complete.
        """

        # Wait until the speech is recognized
        while not self._recognition_finished:
            sleep(1)

        """
        The program sleeps in a loop until recognition is finished, ensuring it waits for the result.
        """

        self._recognition_finished = False

        return self._recognized_speech

    """
    This method starts the main loop for listening to speech input.
    """

    def start_speech_recognition(self) -> None:

        """
        The program continuously listens for speech until the stop condition is met. It captures audio with a specified phrase time limit.
        """
        print("Listening:")
        while not self._stop_listening:
            # Listen to a single phrase
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source=self.microphone, phrase_time_limit=170
                )

            """
            After capturing audio, it recognizes the speech in the background. 
            If recognized, it prints the speech and appends it to the full recognized speech.
            """

            # Recognize speech
            recognized_speech = self._recognize_speech_in_background(audio)

            if recognized_speech:
                print(recognized_speech, end=" ", flush=True)
                self.full_recognized_speech += f" {recognized_speech}"

            """
            The program checks if any stop phrases were recognized, which would stop the listening loop.
            """

            # Check for stop phrase
            for stop_phrase in SpeechRecognizer.STOP_PHRASES:
                if stop_phrase in recognized_speech.lower():
                    self._stop_listening = True


        """
        After stopping, if there is any recognized speech, 
        it cleans up the text by removing extra spaces and stop phrases, 
        and then writes the final text to the specified output file.
        """
        # Write everything to the file
        if self.full_recognized_speech:
            # Prepare `full_recognized_speech` for writing
            self.full_recognized_speech.replace("  ", " ")
            for stop_phrase in SpeechRecognizer.STOP_PHRASES:
                self.full_recognized_speech = self.full_recognized_speech.replace(
                    stop_phrase, ""
                )

            # Write to file
            with open(self.output_file_path, "w", encoding="utf-8") as file:
                file.write(self.full_recognized_speech)
