from dotenv import load_dotenv
import os
import json
import base64

from openai import OpenAI



def openrouter_llm_api_call(
    system_prompt: str = "",
    user_prompt : str = "Result of 2+2 ?",
    assistant_prompt: str = "",
    model: str ="openai/gpt-4o",
    env_var_open_router_base_url: str = "OPENAI_BASE_URL", # "OPENROUTER_BASE_URL",
    env_var_open_router_api_key: str = "OPENAI_API_KEY", # "OPENROUTER_API_KEY"
    temperature: float | None = None,
    ):
    """

    Function to execute API calls with Openrouter using the OpenAI chat completions API.
    This function sets up an OpenAI client with OpenRouter credentials and makes a chat completion
    request using the specified model and prompts.

    Basically you need a .env file with the OPENROUTER_BASE_URL and OPENROUTER_API_KEY vars set.

    Args:
        system_prompt (str, optional): The system message to set context. Defaults to "".
        user_prompt (str, optional): The user message/query. Defaults to "Result of 2+2 ?".
        assistant_prompt (str, optional): The assistant's previous message for context. Defaults to "".
        model (str, optional): The model identifier to use. Defaults to "openai/gpt-4o".
        env_var_open_router_base_url (str, optional): Environment variable name for OpenRouter base URL.
            Defaults to "OPENROUTER_BASE_URL".
        env_var_open_router_api_key (str, optional): Environment variable name for OpenRouter API key.
            Defaults to "OPENROUTER_API_KEY".
        temperature (float, optional): Sampling temperature override.
    Returns:
        str: The content of the model's response message.
    Requires:
        - python-dotenv
        - openai
    Environment Variables:
        - OPENROUTER_BASE_URL: The base URL for OpenRouter API
        - OPENROUTER_API_KEY: The API key for OpenRouter authentication
    """

    # Load environment variables from .env
    load_dotenv(override=True)

    # Get base URL and API key
    base_url = os.getenv(env_var_open_router_base_url)
    api_key = os.getenv(env_var_open_router_api_key)

    # Optional user identifier
    username = os.getenv("USER")

    # Initialize OpenAI client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Prepare common arguments
    request_args = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt}
        ]
    }
    if temperature is not None:
        request_args["temperature"] = float(temperature)

    # Conditionally add user ID if it exists
    if username:
        request_args["extra_headers"] = {"HTTP-Referer": username, "X-Title": username}

    # Make the API call
    completion = client.chat.completions.create(**request_args)

    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def openrouter_llm_api_call_with_historic(
    user_key: str,
    project_name: str,
    system_prompt: str = "",
    user_prompt: str = "",
    model: str = "openai/gpt-4o",
    env_var_open_router_base_url: str = "OPENAI_BASE_URL",
    env_var_open_router_api_key: str = "OPENAI_API_KEY",
    history_file_path: str = "historic.json",
    max_history_messages: int = 20
) -> (str, None):
    """
    Make a chat completion call to OpenRouter, storing and retrieving conversation history
    per user/project in a JSON file.

    Args:
        user_key: Identifier for the user (e.g., username or user ID).
        project_name: Identifier for the project under that user.
        system_prompt: Optional system message.
        user_prompt: The user’s new message.
        model: The model identifier to use.
        env_var_open_router_base_url: Environment var for base URL.
        env_var_open_router_api_key: Environment var for API key.
        history_file_path: Path to the JSON file storing histories.
        max_history_messages: Maximum number of messages in the history to keep (oldest dropped).

    Returns:
        assistant_content: The assistant’s response text.
    """

    load_dotenv()
    base_url = os.getenv(env_var_open_router_base_url)
    api_key = os.getenv(env_var_open_router_api_key)
    if not base_url or not api_key:
        raise ValueError("OPENROUTER_BASE_URL or OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Compute the key used in the JSON file
    hist_key = f"{user_key}:{project_name}"

    # Load existing histories JSON (if exists)
    
    if os.path.exists(history_file_path):
        with open(history_file_path, "r", encoding="utf-8") as f:
            all_histories = json.load(f)
    else:
        all_histories = {}

    # Get the history list for this user/project
    history = all_histories.get(hist_key, [])

    # Add system prompt if provided (and if starting fresh or if you want repeat each time)
    if system_prompt:
        # Optionally include system prompt only once or always; here we include each time
        history.append({"role": "system", "content": system_prompt})

    # Add user message
    history.append({"role": "user", "content": user_prompt})

    # Trim if too many messages
    if len(history) > max_history_messages:
        history = history[-max_history_messages:]

    request_args = {
        "model": model,
        "messages": history,
        "user": hist_key  # optional: pass user identifier to the API for abuse tracking etc.
    }

    # Optional extra headers (if you still want them)
    username = os.getenv("USER")
    if username:
        request_args["extra_headers"] = {"HTTP-Referer": username, "X-Title": username}

    completion = client.chat.completions.create(**request_args)
    assistant_content = completion.choices[0].message.content

    # Append assistant response to history
    history.append({"role": "assistant", "content": assistant_content})

    # Trim again, if needed
    if len(history) > max_history_messages:
        history = history[-max_history_messages:]

    # Store back the updated history
    all_histories[hist_key] = history
    # Ensure directory exists

    hist_dir = os.path.dirname(history_file_path)
    if hist_dir:                      # seulement si on a un dossier dans le chemin
        os.makedirs(hist_dir, exist_ok=True)

    with open(history_file_path, "w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2, ensure_ascii=False)

    # Return the assistant answer
    print(assistant_content)
    return assistant_content


def openrouter_llm_api_call_with_pdf(
    pdf_paths: list[str],
    user_prompt: str,
    system_prompt: str = "",
    assistant_prompt: str = "",
    model: str = "openai/gpt-4o",
    env_var_open_router_base_url: str = "OPENAI_BASE_URL",
    env_var_open_router_api_key: str = "OPENAI_API_KEY",
    pdf_engine: str = "auto",  # New parameter for PDF processing engine
    temperature: float | None = None,
):
    """
    Function to execute API calls with OpenRouter including one or multiple PDF documents.
    The PDFs are encoded in base64 and sent using OpenRouter's universal file format.

    Args:
        pdf_paths (list[str]): List of paths to PDF files to include in the request.
        user_prompt (str): The user's text question/query about the PDFs.
        system_prompt (str, optional): The system message to set context. Defaults to "".
        assistant_prompt (str, optional): The assistant's previous message for context. Defaults to "".
        model (str, optional): The model identifier to use. Defaults to "openai/gpt-4o".
        env_var_open_router_base_url (str, optional): Environment variable name for OpenRouter base URL.
            Defaults to "OPENAI_BASE_URL".
        env_var_open_router_api_key (str, optional): Environment variable name for OpenRouter API key.
            Defaults to "OPENAI_API_KEY".
        pdf_engine (str, optional): PDF processing engine. Options: "auto", "native", "mistral-ocr", 
            "unstructured". Defaults to "auto" which uses native support if available.
        temperature (float, optional): Sampling temperature override.

    Returns:
        str: The content of the model's response message.

    Raises:
        FileNotFoundError: If any PDF file doesn't exist.
        ValueError: If environment variables are not set or if pdf_paths is empty.

    Example:
        # Single PDF with Gemini 2.5 Flash
        response = openrouter_llm_api_call_with_pdf(
            pdf_paths=["document.pdf"],
            user_prompt="What are the main findings in this document?",
            system_prompt="You are a helpful document analyst.",
            model="google/gemini-2.5-flash-preview"
        )
        
        # Multiple PDFs with Claude Sonnet 4.5
        response = openrouter_llm_api_call_with_pdf(
            pdf_paths=["report1.pdf", "report2.pdf", "analysis.pdf"],
            user_prompt="Compare the findings across these three reports",
            system_prompt="You are a helpful document analyst.",
            model="anthropic/claude-sonnet-4.5"
        )
        
        # With OCR for scanned documents
        response = openrouter_llm_api_call_with_pdf(
            pdf_paths=["scanned_doc.pdf"],
            user_prompt="Extract all text from this scanned document",
            model="google/gemini-2.5-flash-preview",
            pdf_engine="mistral-ocr"
        )
    """

    # Load environment variables from .env
    load_dotenv()

    # Get base URL and API key
    base_url = os.getenv(env_var_open_router_base_url)
    api_key = os.getenv(env_var_open_router_api_key)

    if not base_url or not api_key:
        raise ValueError(f"{env_var_open_router_base_url} or {env_var_open_router_api_key} environment variable not set")

    # Validate pdf_paths
    if not pdf_paths or not isinstance(pdf_paths, list):
        raise ValueError("pdf_paths must be a non-empty list of PDF file paths")

    # Check if all PDFs exist and encode them using OpenRouter's format
    pdf_documents = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Read and encode PDF to base64
        with open(pdf_path, "rb") as pdf_file:
            pdf_base64 = base64.b64encode(pdf_file.read()).decode('utf-8')
            
            # OpenRouter's universal file format (works with ALL models)
            pdf_documents.append({
                "type": "file",  # ✅ OpenRouter uses "file" not "document"
                "file": {
                    "filename": os.path.basename(pdf_path),
                    "file_data": f"data:application/pdf;base64,{pdf_base64}"  # ✅ Data URL format
                }
            })
    
    print(f"Loaded {len(pdf_documents)} PDF(s) for model: {model}")

    # Optional user identifier
    username = os.getenv("USER")

    # Initialize OpenAI client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Build the messages list
    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add user message with PDF(s) and text
    user_content = []
    
    # Add all PDF documents first
    user_content.extend(pdf_documents)
    
    # Then add the text prompt
    user_content.append({
        "type": "text",
        "text": user_prompt
    })
    
    messages.append({"role": "user", "content": user_content})

    # Add assistant prompt if provided
    if assistant_prompt:
        messages.append({"role": "assistant", "content": assistant_prompt})

    # Prepare request arguments
    request_args = {
        "model": model,
        "messages": messages
    }
    if temperature is not None:
        request_args["temperature"] = float(temperature)

    # Add PDF processing engine configuration if not auto
    if pdf_engine != "auto":
        request_args["extra_body"] = {
            "plugins": [
                {
                    "id": "file-parser",
                    "pdf": {
                        "engine": pdf_engine
                    }
                }
            ]
        }

    # Conditionally add user ID if it exists
    if username:
        print(f"User: {username}")
        request_args["extra_headers"] = {
            "HTTP-Referer": username, 
            "X-Title": username
        }

    # Make the API call
    print(f"Calling OpenRouter with {len(pdf_paths)} PDF(s)...")
    completion = client.chat.completions.create(**request_args)

    response_content = completion.choices[0].message.content
    print(response_content)
    return response_content



if __name__ == "__main__":
    # Example usage
    openrouter_llm_api_call(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is the capital of Madagascar?"
    )
