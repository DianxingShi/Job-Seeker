import json
import numpy as np
from scipy.stats import pearsonr
import math


json_filename = "your jsons file"
answers_filename = "your answers file"


green_question_numbers = [
    5, 7, 8, 16, 17, 20, 26, 27, 39, 44, 45, 48, 51, 53, 65, 68, 80, 89, 94, 
    113, 116, 125, 126, 129, 134, 136, 149, 157, 159, 164, 173, 176, 185, 
    186, 190, 196, 199, 201, 202, 212, 215
]
red_question_numbers = [
    205, 174, 167, 168, 156, 154, 148, 131, 124, 117, 114, 107, 103, 50, 
    36, 33, 3
]


focused_question_numbers = green_question_numbers + red_question_numbers


try:
    
    with open(json_filename, 'r', encoding='utf-8') as f:
        user_data = json.load(f)
    user_answers = [int(item['answer']) for item in user_data]
    

    
    with open(answers_filename, 'r') as f:
        standard_answers = [int(line.strip()) for line in f if line.strip()]
    
    

    
    if len(user_answers) != len(standard_answers):
        print("\n ERROR")
        print(f"Number of your answers: {len(user_answers)}")
        print(f"Number of reference answers: {len(standard_answers)}")
    else:
        
        user_answers_np = np.array(user_answers)
        standard_answers_np = np.array(standard_answers)

        
        mse_all = np.mean((user_answers_np - standard_answers_np) ** 2)
        rmse_all = math.sqrt(mse_all)
        correlation_all, _ = pearsonr(user_answers_np, standard_answers_np)

        
        focused_indices = np.array([num - 1 for num in focused_question_numbers])
        focused_user_answers = user_answers_np[focused_indices]
        focused_standard_answers = standard_answers_np[focused_indices]
        mse_focused = np.mean((focused_user_answers - focused_standard_answers) ** 2)
        rmse_focused = math.sqrt(mse_focused)
        correlation_focused, _ = pearsonr(focused_user_answers, focused_standard_answers)

        
        green_indices = np.array([num - 1 for num in green_question_numbers])
        red_indices = np.array([num - 1 for num in red_question_numbers])
        
        
        avg_standard_green = np.mean(standard_answers_np[green_indices])
        avg_standard_red = np.mean(standard_answers_np[red_indices])
        
        
        avg_user_green = np.mean(user_answers_np[green_indices])
        avg_user_red = np.mean(user_answers_np[red_indices])

        

        print(f"RMSE: {rmse_all:.4f}")
        print(f"Pearson Correlation: {correlation_all:.4f}")

 
        print(f"Number(P+N): {len(focused_indices)}")
        print(f"RMSE(P+N): {rmse_focused:.4f}")
        print(f"Pearson Correlation(P+N): {correlation_focused:.4f}")

        print("\n--- Averages(P+N) ---")
        print(f"Reference -> postive ({len(green_indices)}道) Avg: {avg_standard_green:.4f}")
        print(f"Reference -> negetive ({len(red_indices)}道) Avg: {avg_standard_red:.4f}")
        print(f"Reference -> postive ({len(green_indices)}道) Avg: {avg_user_green:.4f}")
        print(f"Reference -> negetive ({len(red_indices)}道) Avg: {avg_user_red:.4f}")


except FileNotFoundError as e:
    print(f"\n ERROR in finding files: {e}")
except Exception as e:
    print(f"\n ERROR in processing: {e}")