import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import time
import re

from openai import OpenAI, AzureOpenAI

# Azure AI Inference 
# pip install azure-ai-inference azure-identity
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from google import genai
from google.genai import types

# --- App and Directory Setup ---
app = Flask(__name__)
SAVE_DIR = "candidate_conversations"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


MODEL_CONFIGS = {
    "qwen1.5-32b-chat": {
        "api_type": "openai_compatible",
        "client_config": {
            "api_key": os.getenv("DASHSCOPE_API_KEY", "your API"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        "model_name": "qwen1.5-32b-chat"
    },
    "qwen2.5-32b-instruct": {
        "api_type": "openai_compatible",
        "client_config": {
            "api_key": os.getenv("DASHSCOPE_API_KEY", "your API"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        "model_name": "qwen2.5-32b-instruct"
    },
    "qwen3-32b": {
        "api_type": "openai_compatible",
        "client_config": {
            "api_key": os.getenv("DASHSCOPE_API_KEY", "your API"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        "model_name": "qwen3-32b",
         "extra_params": {
            "enable_thinking": False
        }
    },
    
    "gpt-4o": {
        "api_type": "azure",
        "client_config": {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY", "your API"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "your ENDPOINT"),
            "api_version": "2025-01-01-preview",
        },
        "model_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    },
    "gpt-5-chat": {
        "api_type": "azure",
        "client_config": {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY", "your API"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "your ENDPOINT"),
            "api_version": "2025-01-01-preview",
        },
        "model_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-chat")
    },
    
    # azure-ai-inference SDK  ---
   "deepseek-r1": {
        "api_type": "openai_compatible",
        "client_config": {
            "api_key": os.getenv("DASHSCOPE_API_KEY", "your API"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        "model_name": "deepseek-r1"
    },
    "deepseek-v3.1": {
        "api_type": "azure_inference",
        "client_config": {
            "endpoint": os.getenv("AZURE_INFERENCE_ENDPOINT", "your ENDPOINT"),
            "key": os.getenv("AZURE_INFERENCE_KEY", "your API")
        },
        "model_name": "DeepSeek-V3.1" 
    },
    "deepseek-v3.2-exp": {
        "api_type": "openai_compatible",
        "client_config": {
            "api_key": os.getenv("DASHSCOPE_API_KEY", "your API"),
            "base_url": "https://api.deepseek.com",
        },
        "model_name": "deepseek-chat"
    },
    "deepseek-v3.1-terminus": {
        "api_type": "openai_compatible",
        "client_config": {
            "api_key": os.getenv("DASHSCOPE_API_KEY", "your API"),
            "base_url": "https://api.deepseek.com/v3.1_terminus_expires_on_20251015",
        },
        "model_name": "deepseek-chat"
    },
    
    "gemini2.5-flash": {
        "api_type": "gemini",
        "client_config": {
            
            "api_key": os.getenv("GEMINI_API_KEY", "your API")
        },
        "model_name": "gemini-2.5-flash"
    },
    "gemini2.5-pro": {
        "api_type": "gemini",
        "client_config": {
            
            "api_key": os.getenv("GEMINI_API_KEY", "your API")
        },
        "model_name": "gemini-2.5-pro"
    },

    "llama4-maverick-17b-128e-instruct-fp8": {
        "api_type": "azure_inference",
        "client_config": {
            "endpoint": os.getenv("AZURE_INFERENCE_ENDPOINT", "your ENDPOINT"),
            "key": os.getenv("AZURE_INFERENCE_KEY", "your API")
        },
        "model_name": "Llama-4-Maverick-17B-128E-Instruct-FP8" 
    },
    "llama-3.3-70b-instruct": {
        "api_type": "azure_inference",
        "client_config": {
            "endpoint": os.getenv("AZURE_INFERENCE_ENDPOINT", "your ENDPOINT"),
            "key": os.getenv("AZURE_INFERENCE_KEY", "your API")
        },
        "model_name": "Llama-3.3-70B-Instruct" 
    },
}

# --- In-memory storage for the current session ---
session_state = {
    "client": None,
    "model_name": None,
    "api_type": None, 
    "extra_params": None,
    "conversation_history": [],
    "current_language": "zh"
}

@app.route('/')
def index():
    return render_template('candidate.html')

@app.route('/start_task', methods=['POST'])
def start_task():
    global session_state
    data = request.json
    task_description = data.get('task_description')
    selected_model_key = data.get('model')
    session_state['current_language'] = data.get('language', 'zh')

    config = MODEL_CONFIGS.get(selected_model_key)
    if not config:
        return jsonify({"error": f"Invalid model key: {selected_model_key}"}), 400

    try:
        api_type = config["api_type"]
        client_params = config["client_config"]
        session_state["api_type"] = api_type 
        session_state["extra_params"] = config.get("extra_params")

        if api_type == "azure":
            session_state["client"] = AzureOpenAI(**client_params)
        elif api_type == "openai_compatible":
            session_state["client"] = OpenAI(**client_params)
        elif api_type == "azure_inference": 
            session_state["client"] = ChatCompletionsClient(
                endpoint=client_params["endpoint"],
                credential=AzureKeyCredential(client_params["key"])
            )
        elif api_type == "gemini":
            api_key = client_params.get("api_key")
            if not api_key or "YOUR_GEMINI_API_KEY_HERE" in api_key:
               raise ValueError("GEMINI_API_KEY is not set. Please set it as an environment variable.")
            os.environ['GEMINI_API_KEY'] = api_key
            session_state["client"] = genai.Client()
        else:
            return jsonify({"error": f"Unsupported API type: {api_type}"}), 500
            
        session_state["model_name"] = config["model_name"]

    except Exception as e:
        return jsonify({"error": f"Failed to initialize AI client: {str(e)}"}), 500

    lang_instruction = {'zh': '\n\n重要：请用中文回答所有问题。', 'en': '\n\nImportant: Please answer all questions in English.'}.get(session_state['current_language'], '')
    full_task_description = task_description + lang_instruction
    session_state['conversation_history'] = [{"role": "system", "content": full_task_description}]

    response_message = {'zh': f"任务已设定，使用 {selected_model_key} 模型。可以开始提问。", 'en': f"Task has been set using the {selected_model_key} model. You can start asking questions."}
    return jsonify({"status": "success", "message": response_message.get(session_state['current_language'])})


def get_gpt_response(question):
    global session_state
    client = session_state['client']
    model = session_state['model_name']
    history = session_state['conversation_history']
    lang = session_state['current_language']
    api_type = session_state['api_type']
    gen_config_params = session_state.get('generation_config', {})
    extra_params = session_state.get('extra_params', {})
    history.append({"role": "user", "content": question})

    try:
        gpt_answer = ""
        
        if api_type == "azure_inference":
            
            messages_for_inference = []
            for msg in history:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    messages_for_inference.append(SystemMessage(content=content))
                elif role == "user":
                    messages_for_inference.append(UserMessage(content=content))
                elif role == "assistant":
                    messages_for_inference.append(AssistantMessage(content=content))
            
            
            response = client.complete(messages=messages_for_inference, model=model, max_tokens=1600)
            
            if response.choices:
                gpt_answer = response.choices[0].message.content
            # time.sleep(5.0)
        elif api_type == "gemini":
            gemini_history = []
            for msg in history:
                # This client does not have a distinct 'system' role.
                # Treat 'system' as the first 'user' message.
                role = "model" if msg["role"] == "assistant" else "user"
                # The 'parts' format must be a list of dicts with a 'text' key
                gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
            thinking_config = types.ThinkingConfig(thinking_budget=512)
            generation_config = types.GenerateContentConfig(thinking_config=thinking_config)
            response = client.models.generate_content(
                model=f"models/{model}",
                contents=gemini_history,
                config=generation_config
            )
            # This client often returns text directly in the 'text' attribute
            gpt_answer = response.text
            # time.sleep(3.0)
        else: # for "azure" and "openai_compatible"
            
            api_call_params = {
    "model": model,
    "messages": history,
    "max_tokens": 2000
}
            if extra_params:
                api_call_params["extra_body"] = extra_params 
            for attempt in range(6):
                try:
                    response = client.chat.completions.create(**api_call_params)
                    break
                except Exception as e:
                    if "limit_requests" in str(e) or "429" in str(e):
                        time.sleep(2 * (attempt + 1))
                        continue
                    raise e
            else:
                raise Exception("QWEN API limit, retry failed.")
            if response.choices:
                gpt_answer = response.choices[0].message.content
        
        history.append({"role": "assistant", "content": gpt_answer})
        return gpt_answer

    except Exception as e:
        history.pop()
        error_messages = {'zh': f"调用API时出错: {str(e)}", 'en': f"API call error: {str(e)}"}
        return error_messages.get(lang, str(e))



@app.route('/ask', methods=['POST'])
def ask():
    
    global session_state
    lang = session_state['current_language']

    if not session_state.get('client'):
        error_messages = {'zh': "请先设定任务和模型", 'en': "Please set the task and model first"}
        return jsonify({"error": error_messages.get(lang)}), 400

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if file and file.filename.endswith('.txt'):
            try:
                questions = file.stream.read().decode('utf-8').splitlines()
                responses = []
                for q in questions:
                    if q.strip():
                        response_text = get_gpt_response(q)
                        responses.append({"question": q, "answer": response_text})
                return jsonify({"responses": responses})
            except Exception as e:
                error_messages = {'zh': f"处理文件时出错: {str(e)}", 'en': f"Error processing file: {str(e)}"}
                return jsonify({"error": error_messages.get(lang, str(e))}), 500
        else:
            error_messages = {'zh': "请上传.txt格式的文件", 'en': "Please upload a .txt file"}
            return jsonify({"error": error_messages.get(lang)}), 400

    data = request.json
    user_question = data.get('question')
    if not user_question:
        error_messages = {'zh': "问题不能为空", 'en': "Question cannot be empty"}
        return jsonify({"error": error_messages.get(lang)}), 400

    answer = get_gpt_response(user_question)
    return jsonify({"answer": answer})

@app.route('/save_conversation', methods=['POST'])
def save_conversation():
    
    global session_state
    lang = session_state['current_language']
    history = session_state['conversation_history']
    model_name = session_state.get('model_name', 'unknown_model')
    if len(history) <= 1:
        messages = {'zh': "没有问答记录，无需保存。", 'en': "No Q&A records to save."}
        return jsonify({"status": "info", "message": messages.get(lang)})

    safe_model_name = model_name.replace("/", "_").replace(":", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"conversation_{timestamp}_{safe_model_name}.json")

    qa_pairs = []
    for i in range(1, len(history), 2):
        if i + 1 < len(history):
            qa_pairs.append({
                "question": history[i]['content'],
                "answer": history[i+1]['content']
            })
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
        
        session_state['conversation_history'] = history[:1]
        messages = {
            'zh': f"对话已成功保存至 {filename}，AI记忆已重置。",
            'en': f"Conversation saved to {filename} and AI memory has been reset."
        }
        return jsonify({"status": "success", "message": messages.get(lang)})
    except Exception as e:
        error_messages = {'zh': f"保存文件时出错: {str(e)}", 'en': f"Error saving file: {str(e)}"}
        return jsonify({"error": error_messages.get(lang, str(e))}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)