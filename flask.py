import openai
from textblob import TextBlob
from flask import Flask, request, jsonify
import requests
import re

app = Flask(__name__)

from flask import send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'github_analysis.html')


def preprocess_code_files(files):
    token_limit = 4096
    processed_files = []

    for file in files:
        file_name = file["name"]
        file_url = file["download_url"]

        response = requests.get(file_url)
        if response.status_code == 200:
            file_content = response.text

            # Check if file content exceeds token limit
            if len(file_content.split()) > token_limit:
                if file_name.endswith('.ipynb'):
                    # Process large Jupyter notebooks
                    processed_content = preprocess_large_notebook(file_content)
                    processed_files.append((file_name, processed_content))
                else:
                    # Process other large files (truncation)
                    truncated_content = file_content[:token_limit]
                    processed_files.append((file_name, truncated_content))
            else:
                processed_files.append((file_name, file_content))
        else:
            print(f"Error fetching file: {file_name}")

    return processed_files

def preprocess_large_notebook(notebook_content):
    # Split the notebook into smaller parts (code and text cells)
    parts = get_ipython().display_formatter.format(notebook_content, 'notebook')[0]['text/plain']
    token_limit = 4096  # Define token limit here
    processed_content = ""
    for part in parts:
        if len(processed_content) + len(part) <= token_limit:
            processed_content += part
        else:
            break
    return processed_content

max_context_length = 400
def generate_prompt_response(prompt):
    openai.api_key = "GIVE API KEY"  # Replace with your actual OpenAI API key

    # Calculate the maximum available tokens for completion
    max_prompt_tokens = max_context_length // 2  # Allocate half of the context length to the prompt
    max_completion_tokens = max_context_length - max_prompt_tokens

    # Truncate prompt if it's too long
    if len(prompt.split()) > max_prompt_tokens:
        prompt = " ".join(prompt.split()[:max_prompt_tokens])

    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose an appropriate GPT-3 engine
        prompt=prompt,
        max_tokens=max_completion_tokens,
        temperature=0.7,  # Adjust the temperature for creativity
        stop=None  # You can add a stop phrase if needed
    )
    return response.choices[0].text.strip()

def evaluate_technical_complexity(code_content):
    prompt = f"Evaluate the technical complexity of the following code:\n\n{code_content}\n\nTechnical complexity:"
    response = generate_prompt_response(prompt)
    return response

def extract_sentiment_score(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

def fetch_repo_contents(owner, repo_name, access_token):
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents"
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        repo_contents = response.json()
        return repo_contents
    else:
        print(f"Error fetching repository contents: {response.status_code}")
        return None
def extract_username_from_url(url):
    match = re.match(r"https://github\.com/([A-Za-z0-9_-]+)/?", url)
    if match:
        return match.group(1)
    return None
def fetch_user_repositories(username, access_token):
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    api_url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        repositories = response.json()
        return repositories
    else:
        print(f"Error fetching repositories: {response.status_code}")
        return None

def justify_repository_selection(repo_name, complexity_justification):
    prompt = f"Justify the selection of the '{repo_name}' repository as the most technically complex repository:\n\n{complexity_justification}\n\nJustification:"
    response = generate_prompt_response(prompt)
    return response

@app.route('/analyze')
def analyze():
    github_url = request.args.get('githubUrl')

    # Replace with your GitHub access token
    access_token = "GIVE YOUR ACCESS TOKEN"
    username = extract_username_from_url(github_url)
    repositories = fetch_user_repositories(username, access_token)
    if repositories:
        most_complex_repository = None
        highest_complexity_score = float('-inf')
        most_complex_justification = ""
        for repo in repositories:
            owner = repo["owner"]["login"]
            repo_name = repo["name"]
            repo_contents = fetch_repo_contents(owner, repo_name, access_token)

            if repo_contents:
                processed_files = preprocess_code_files(repo_contents)

                repo_complexity_score = 0
                repo_justification = ""

                for file_name, file_content in processed_files:
                    complexity_response = evaluate_technical_complexity(file_content)
                    complexity_score = extract_sentiment_score(complexity_response)

                    repo_complexity_score += complexity_score
                    repo_justification += f"File '{file_name}': {complexity_response}\n"

                if repo_complexity_score > highest_complexity_score:
                    highest_complexity_score = repo_complexity_score
                    most_complex_repository = repo_name
                    most_complex_justification = repo_justification

        if most_complex_repository:
            repository_name = most_complex_repository
            repository_link = "https://github.com/"+username+"/"+ most_complex_repository
            gpt_analysis = justify_repository_selection(most_complex_repository, most_complex_justification)

            return jsonify({
                "repositoryName": repository_name,
                "repositoryLink": repository_link,
                "gptAnalysis": gpt_analysis
            })
        else:
            return jsonify({"message": "No complex repositories found."})

if __name__ == '__main__':
    app.run(debug=True)
