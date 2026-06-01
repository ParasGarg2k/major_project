import os

# Global variable: folder containing the .txt files
TXT_FOLDER = r'D:\Artificial Intelligence\IIT Mandi Internship\MTP_final\Final_MSTCN_repo\data\new_4_tasks_dataset_official_approach\ground_truth'  # ← Replace with your folder path

incorrect_word = "Plave"
correct_word = "Place"

COUNTER = 0

# Iterate through all .txt files in the folder
for filename in os.listdir(TXT_FOLDER):
    if filename.endswith('.txt'):
        file_path = os.path.join(TXT_FOLDER, filename)

        # Read and correct lines
        with open(file_path, 'r') as f:
            lines = f.readlines()

        corrected_lines = []
        for line in lines:
            word = line.strip()
            if word == incorrect_word:
                corrected_lines.append(correct_word + "\n")
                COUNTER += 1
            else:
                corrected_lines.append(word + '\n')

        # Overwrite the file with corrected lines
        with open(file_path, 'w') as f:
            f.writelines(corrected_lines)

print(f"{COUNTER} .txt files processed. '{incorrect_word}' has been replaced with '{correct_word}'.")
