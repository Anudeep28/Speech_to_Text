import kenlm

# Specify the path to your text file
text_file = "path/to/your/text_corpus.txt"

# Set the output path for the ARPA format model
arpa_model_path = "path/to/your/model.arpa"

# Set the output path for the binary format model
binary_model_path = "path/to/your/model.binary"

# Train the model using lmplz
!kenlm/build/bin/lmplz -o 5 --text {text_file} --arpa {arpa_model_path}

# Convert the ARPA model to binary format for faster loading
!kenlm/build/bin/build_binary -q 8 -b 7 -a 256 trie {arpa_model_path} {binary_model_path}