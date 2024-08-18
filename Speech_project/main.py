from recognition import SpeechRecognizer


def main():
    speech_recognizer = SpeechRecognizer(
        output_file_path="./recognized_text.txt",
        whisper_model="large-v3",
    )
    speech_recognizer.start_speech_recognition()

if __name__ == "__main__":
    main()
