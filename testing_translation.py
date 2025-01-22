import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Load the tokenizer and model
model_path = "english_to_hindi_translator"  # Update the model directory if different

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

# Function to translate English text to Hindi using beam search
def translate_text_with_beam_search(input_text: str, max_length: int = 100, num_beams: int = 5):
    # Tokenize the input text
    tokenized_input = tokenizer([input_text], return_tensors="np")

    # Generate translation with beam search
    output_tokens = model.generate(
        **tokenized_input,
        max_length=max_length,  # Set maximum length for output
        num_beams=num_beams,  # Use beam search with the specified number of beams
        early_stopping=True,  # Stop when the best hypothesis is found
        no_repeat_ngram_size=2  # Avoid repeating n-grams in the output
    )

    # Decode the output tokens
    translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return translated_text

# Test the model with beam search
input_sentence = "I ate water and read a book."
translated_sentence = translate_text_with_beam_search(input_sentence, num_beams=5)

print(f"Input: {input_sentence}")
print(f"Translated (Beam Search): {translated_sentence}")

# Save the input and translated output to a text file
output_file = "translation_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Input: {input_sentence}\n")
    f.write(f"Translated: {translated_sentence}\n")

print(f"Translation saved to {output_file}")
