import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# --- Dependency Imports (Copied from candidate_test.py) ---
# Types 1 & 2: OpenAI / Azure OpenAI
from openai import OpenAI, AzureOpenAI

# Type 3: Azure AI Inference
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

# Type 4: Google Gemini
from google import genai
from google.genai import types

# --- App and Directory Setup ---
app = Flask(__name__)
SAVE_DIR = "hr_analyses"
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
            # Ensure the GEMINI_API_KEY environment variable is set.
            "api_key": os.getenv("GEMINI_API_KEY", "your API")
        },
        "model_name": "gemini-2.5-flash"
    },
    "gemini2.5-pro": {
        "api_type": "gemini",
        "client_config": {
            # Ensure the GEMINI_API_KEY environment variable is set.
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
    "current_language": "zh",
    "last_filename": None,
    "last_analysis": None
}

@app.route('/')
def index():
    """Render the main HR page"""
    return render_template('hr.html')

@app.route('/start_task', methods=['POST'])
def start_task():
    """Initialize the AI client based on the selected model and set the system prompt."""
    global session_state
    data = request.json
    task_description = data.get('task_description')
    selected_model_key = data.get('model')
    session_state['current_language'] = data.get('language', 'zh')

    if not task_description:
        return jsonify({"error": "Task description cannot be empty"}), 400
    if not selected_model_key:
        return jsonify({"error": "Model not selected"}), 400

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
               raise ValueError("GEMINI_API_KEY is not set.")
            os.environ['GEMINI_API_KEY'] = api_key
            session_state["client"] = genai.Client()
        else:
            return jsonify({"error": f"Unsupported API type: {api_type}"}), 500

        session_state["model_name"] = config["model_name"]

    except Exception as e:
        return jsonify({"error": f"Failed to initialize AI client: {str(e)}"}), 500

    lang_instruction = {
        'zh': '\n\n重要：请用中文回答。',
        'en': '\n\nImportant: Please answer in English.'
    }.get(session_state['current_language'], '')
    full_task_description = task_description + lang_instruction

    session_state['conversation_history'] = [{"role": "system", "content": full_task_description}]

    response_message = {
        'zh': f"HR角色已使用 {selected_model_key} 模型设定，请上传求职者作答文件进行分析。",
        'en': f"HR role has been set with {selected_model_key} model. Please upload the candidate's answer file for analysis."
    }
    return jsonify({"status": "success", "message": response_message.get(session_state['current_language'])})


@app.route('/analyze_answers', methods=['POST'])
def analyze_answers():
    """Process the uploaded candidate's answer file and get the AI's judgment."""
    global session_state
    lang = session_state['current_language']

    if not session_state.get('client'):
        error_messages = {'zh': "请先设定HR角色和模型", 'en': "Please set the HR role and model first"}
        return jsonify({"error": error_messages.get(lang)}), 400

    if 'file' not in request.files or request.files['file'].filename == '':
        error_messages = {'zh': "未找到文件", 'en': "No file found"}
        return jsonify({"error": error_messages.get(lang)}), 400

    file = request.files['file']
    if not file.filename.endswith('.json'):
        error_messages = {'zh': "请上传 .json 格式的文件", 'en': "Please upload a .json file"}
        return jsonify({"error": error_messages.get(lang)}), 400

    try:
        file_content = file.stream.read().decode('utf-8')
        qa_pairs = json.loads(file_content)

        qa_text = "".join([f"Question {i+1}: {pair.get('question', 'N/A')}\nAnswer {i+1}: {pair.get('answer', 'N/A')}\n\n" for i, pair in enumerate(qa_pairs)])
        
        prompt_templates = {
            'zh': f"你将基于之前设定的HR角色，分析以下候选人的问答内容。请给出专业、全面、有建设性的评估和判断。\n\n--- 候选人问答 ---\n{qa_text}--- 分析评估 ---\n",
            'en': f"Based on the HR role you have been assigned, please analyze the following Q&A from a candidate. Provide a professional, comprehensive, and constructive evaluation and judgment.\n\n--- Candidate's Q&A ---\n{qa_text}--- Analysis & Evaluation ---\n"
        }
        user_prompt = prompt_templates.get(lang, prompt_templates['en'])
        
      
        client = session_state['client']
        model = session_state['model_name']
        api_type = session_state['api_type']
        extra_params = session_state.get('extra_params', {})
        system_prompt = session_state['conversation_history'][0]
        messages_for_api = [system_prompt, {"role": "user", "content": user_prompt}]

        gpt_analysis = ""
        if api_type == "azure_inference":
            messages_for_inference = [SystemMessage(content=system_prompt['content']), UserMessage(content=user_prompt)]
            response = client.complete(messages=messages_for_inference, model=model, max_tokens=2048)
            if response.choices:
                gpt_analysis = response.choices[0].message.content
        elif api_type == "gemini":
            gemini_history = [{"role": "user", "parts": [{"text": system_prompt['content']}]}, {"role": "model", "parts": [{"text": "OK."}]}, {"role": "user", "parts": [{"text": user_prompt}]}]
            response = client.models.generate_content(model=f"models/{model}", contents=gemini_history)
            gpt_analysis = response.text
        else: # "azure" and "openai_compatible"
            api_call_params = {"model": model, "messages": messages_for_api, "max_tokens": 2000}
            if extra_params:
                api_call_params["extra_body"] = extra_params
            response = client.chat.completions.create(**api_call_params)
            if response.choices:
                gpt_analysis = response.choices[0].message.content

        session_state["last_filename"] = file.filename
        session_state["last_analysis"] = gpt_analysis

        return jsonify({"analysis": gpt_analysis})

    except json.JSONDecodeError:
        return jsonify({"error": {'zh': "文件内容不是有效的JSON格式。", 'en': "The file content is not valid JSON."}.get(lang)}), 400
    except Exception as e:
        return jsonify({"error": {'zh': f"处理文件或调用API时出错: {str(e)}", 'en': f"Error processing file or calling API: {str(e)}"}.get(lang)}), 500


@app.route('/save_analysis', methods=['POST'])
def save_analysis():
    """Save the last analysis to a JSON file."""
    global session_state
    model_name = session_state.get('model_name', 'unknown_model')
    lang = request.json.get('language', session_state['current_language'])

    if not session_state.get("last_filename") or not session_state.get("last_analysis"):
        messages = {'zh': "没有可保存的分析记录。", 'en': "No analysis record to save."}
        return jsonify({"status": "info", "message": messages.get(lang)})

    safe_model_name = model_name.replace("/", "_").replace(":", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    new_filename = os.path.join(SAVE_DIR, f"analysis_{timestamp}_{safe_model_name}.json")

    data_to_save = {
        "source_file": session_state["last_filename"],
        "analysis_content": session_state["last_analysis"]
    }

    try:
        with open(new_filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        
        session_state["last_filename"] = None
        session_state["last_analysis"] = None

        messages = {'zh': f"分析已成功保存至 {new_filename}。", 'en': f"Analysis successfully saved to {new_filename}."}
        return jsonify({"status": "success", "message": messages.get(lang)})
    except Exception as e:
        error_messages = {'zh': f"保存文件时出错: {str(e)}", 'en': f"Error saving file: {str(e)}"}
        return jsonify({"error": error_messages.get(lang)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)