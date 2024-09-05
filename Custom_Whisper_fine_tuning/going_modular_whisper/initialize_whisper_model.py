import whisper
from whisper.tokenizer import get_tokenizer



def initialize_whisper(model_type: str="tiny.en",
                       multilingual: bool=False,
                       language: str="en",
                       task: str="transcribe",
                       device: str="cpu"):
    """
    Initializes a Whisper model and tokenizer for transcription tasks.
    
    model_type: Size	Parameters	English-only model	Multilingual model	Required VRAM	Relative speed
                tiny	   39 M	      tiny.en	            tiny	~1 GB	~32x
                base	   74 M	      base.en	            base	~1 GB	~16x
                small	   244 M	  small.en	        small	~2 GB	~6x
                medium	   769 M	  medium.en	        medium	~5 GB	~2x
                large	   1550 M	  N/A	large	                ~10 GB	 1x
    
    Multilingual -> Multiple Languages

    language: LANGUAGES = {
                            "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
                            "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
                            "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
                            "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
                            "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
                            "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
                            "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
                            "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
                            "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
                            "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
                            "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
                            "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
                            "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
                            "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
                            "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
                            "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
                            "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans",
                            "oc": "occitan", "ka": "georgian", "be": "belarusian", "tg": "tajik", "sd": "sindhi",
                            "gu": "gujarati", "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek",
                            "fo": "faroese", "ht": "haitian creole", "ps": "pashto", "tk": "turkmen",
                            "nn": "nynorsk", "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
                            "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese", "tt": "tatar",
                            "haw": "hawaiian", "ln": "lingala", "ha": "hausa", "ba": "bashkir", "jw": "javanese",
                            "su": "sundanese", "yue": "cantonese",
                        }
    task: translate or transcribe

    """
    # Initialize the Whisper model
    model = whisper.load_model(model_type).to(device)


    # Get the token vocabulary from the Whisper model
    tokenizer = get_tokenizer(multilingual=multilingual, language=language, task=task)

    # To get the start of transcript token id for preprocesssing
    decoder_start_token_id = tokenizer.sot

    return model, decoder_start_token_id