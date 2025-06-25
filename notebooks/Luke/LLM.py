import subprocess
import re
import time
import os
from CommonsenseDataset import CommonsenseDataset

os.makedirs("llm_logs", exist_ok=True)
log_file = open(f"llm_logs/log{int(time.time())}.txt", "w+")
def log(message=""):
  message = str(message)
  message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
  log_file.write(message + "\n")
  print(message)
  

MODEL = "deepseek-r1:32b"

def prompt_model(prompt: str):
  try:
    result = subprocess.run(
      ["ollama", "run", MODEL, prompt],
      capture_output=True,
      text=True,
      encoding="utf-8",
    ).stdout
  except Exception as e:
    log(f"Error running model: {e}")
    return ""
  
  result = result.split("</think>")[-1].strip()
  
  return result

dataset = CommonsenseDataset("./commonsense_qa/data/validation.parquet")

total_questions = 10
correct_answers = 0

for i, data in enumerate(dataset):
  if i >= total_questions:
    break
  
  question = data["question"]
  choices = data["choices"]
  labels = data["labels"]
  
  log(f"Question: {question}")
  
  prompt = f"""
Answer the following question. ONLY REPLY WITH THE NUMBER OF THE CORRECT ANSWER. KEEP THE THINKING TO A MINIMUM. THERE IS ONLY ONE CORRECT ANSWER.

QUESTION:
{question}

CHOICES:
1. {choices[0]}
2. {choices[1]}
3. {choices[2]}
4. {choices[3]}
5. {choices[4]}

REPLY WITH THE NUMBER OF THE CORRECT ANSWER. DO NOT REPLY WITH ANY OTHER TEXT.
"""
  
  result = prompt_model(prompt)
  
  single_digits = list(set([x[1] for x in re.findall(r"(^|\D)([1-5])(\D|$)", result)]))
  
  if len(single_digits) == 1:
    result = int(single_digits[0])
    
    if labels[result - 1]:
      log(f"Correct: {choices[result - 1]}")
      correct_answers += 1
    else:
      log(f"Incorrect: {choices[result - 1]} | {choices}")
  else:
    result = result.replace('\n', '\\n')
    log(f"Invalid result: '{result}'")
    
log(f"Total accuracy: {correct_answers / total_questions * 100:.2f}%")

log_file.close()