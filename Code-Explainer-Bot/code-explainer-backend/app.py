from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
  # Assuming ChatGroq is the correct module

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React frontend

# Initialize ChatGroq with your API key and model name
llm = ChatGroq(
    temperature=0,
    groq_api_key='your_key',
    model_name="llama-3.3-70b-versatile"
)

# Route for the root URL (optional)
@app.route('/')
def index():
    return "Welcome to the Code Explainer API!"

# Route to handle favicon.ico (optional)
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content to avoid 404 errors

# Function to interact with ChatGroq model for code explanation
def explain_code_with_chatgroq(code, language="python"):
    # Construct the prompt to explain the code
    
    prompt = f"""
You are a helpful assistant that explains {language} code line by line in a precise, fixed format.

### Instructions:
For each line of code:
- Start with 'Line X: [original code]'
- Then write 'Explanation:' followed by a clear and simple explanation.
- If there is an error in the line, add 'Corrected:' and provide the fixed version.
- Do not mention what the code does overall.
- Do not include any extra titles like 'Code Explanation', 'Output', or 'Summary'.
- Stick strictly to the format shown in the example.

### Example:
Code:
for i in range(5):
    print(i)

Expected Response:
Line 1: for i in range(5):
Explanation: This starts a loop that runs 5 times, with 'i' taking values from 0 to 4.

Line 2: print(i)
Explanation: This prints the current value of 'i' in each iteration of the loop.

### Code to Explain:
{code}

### Response:
"""





    # Call the ChatGroq model to generate an explanation
    response = llm.invoke(prompt)
    return response.content


# Define the endpoint to explain code
@app.route('/explain', methods=['POST'])
def explain():
    # Get the JSON data from the request
    data = request.get_json()
    code = data.get('code', '')
    language = data.get('language', 'python')

    # If no code is provided, return an error
    if not code.strip():
        return jsonify({'error': 'Code input is empty'}), 400

    try:
        # Get the explanation of the code using the ChatGroq model
        explanation = explain_code_with_chatgroq(code, language)
        return jsonify({'explanation': explanation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
