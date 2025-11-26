from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import pandas as pd
import time
from tqdm import tqdm  # For the progress bar
import diskcache as dc  # For persistent caching

# 1. Setup Persistent Cache
# This creates a folder named ".cache" to store results permanently
dir_path = os.path.dirname(os.path.realpath(__file__))

cache = dc.Cache(f"{dir_path}/.cache")

llm = ChatOllama(base_url="http://172.16.100.50:11434", model="qwen3:8b")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates English to {language}.",
        ),
        (
            "user",
            "please translate the following sentence to {language}: {text} without any other text or explanation.",
        ),
    ]
)

chain = prompt | llm | StrOutputParser()


# 2. Add the @cache.memoize() decorator
# This checks if the input is already in the .cache folder.
# If yes, it returns the saved result INSTANTLY. If no, it runs the AI.
@cache.memoize()
def translate_sentence(sentence: str, language: str):
    # --- NEW FEATURE START ---
    # Check if target language is English (case-insensitive)
    if language.strip().lower() == "english":
        # .isascii() returns True if the string contains only English letters, 
        # numbers, spaces, and punctuation. It returns False for Thai/Chinese/etc.
        if sentence.isascii():
            return sentence
    try:
        response = chain.invoke({"language": language, "text": sentence})

        if "</think>" in response:
            cleaned_response = response.split("</think>")[1].strip()
        else:
            cleaned_response = response.strip()

        return cleaned_response

    except Exception as e:
        return f"[Error] {str(e)}"


if __name__ == "__main__":
    file_name = "plan_control_name.xlsx"
    output_file_name = f"plan_control_name_translated.xlsx"
    output_file_path = os.path.join(dir_path, output_file_name)
    file_path = os.path.join(dir_path, file_name)

    df_dict = pd.read_excel(file_path, sheet_name=None)
    new_df_dict = {}

    # 3. Calculate total for the progress bar
    # We sum up the number of rows in all sheets to give a global estimation
    total_rows = sum(len(df) for df in df_dict.values())

    print("Start translating...")
    start_time = time.time()

    # Create a single progress bar for the whole process
    with tqdm(total=total_rows, unit="sentence") as pbar:

        for sheet_name, df in df_dict.items():
            tqdm.write(f"Processing sheet: {sheet_name}")  # Safe print

            current_df = df.copy()
            sentences = current_df.iloc[:, 1].fillna("").tolist()
            sheet_translations = []

            for i, sentence in enumerate(sentences):
                # Remove 'if i < 2' to run the full process.
                # If you keep it, the progress bar estimation will be wrong.

                # Check for empty sentences to save AI calls
                if not sentence.strip():
                    sheet_translations.append("")
                else:
                    trans = translate_sentence(sentence, "English")
                    sheet_translations.append(trans)

                # Update the progress bar by 1 step
                pbar.update(1)

            # Insert logic
            if "Translated_Column" in current_df.columns:
                current_df["Translated_Column"] = sheet_translations
            elif len(current_df.columns) > 2:
                current_df.insert(2, "Translated_Column", sheet_translations)
            else:
                current_df["Translated_Column"] = sheet_translations

            new_df_dict[sheet_name] = current_df

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")

    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, df in new_df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved to {output_file_path}")
