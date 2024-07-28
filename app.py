from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route('/analyze_cv', methods=['POST'])
def analyze_cv():
    # Get data from the request body
    request_data = request.get_json()
    
    if not request_data:
        return jsonify({"error": "No data provided in the request body"}), 400
    
    if 'sentences' not in request_data:
        return jsonify({"error": "No sentences provided in the request body"}), 400
    
    if 'cv' not in request_data:
        return jsonify({"error": "No CV data provided in the request body"}), 400
    
    sentences_to_compare = request_data['sentences']
    cv_data = request_data['cv']
    
    start_time = time.time()
    
    # Compute embeddings for the sentences
    sentence_embeddings_to_compare = model.encode(sentences_to_compare, convert_to_tensor=True)
    
    # Combine the sentences from 'skill', 'responsibility', 'degree', and 'experience' fields
    combined_sentences = cv_data["skill"] + cv_data["responsibility"] + cv_data["degree"] + cv_data["experience"]
    
    # Compute embeddings
    combined_embeddings = model.encode(combined_sentences, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.cos_sim(sentence_embeddings_to_compare, combined_embeddings)
    
    top_matches = {}
    highest_similarities = []
    
    # Store only the highest match for each comparison sentence
    for idx, comparison_sentence in enumerate(sentences_to_compare):
        sentence_score_pairs = [(combined_sentence, cosine_scores[idx][j].item()) for j, combined_sentence in enumerate(combined_sentences)]
        top_pair = max(sentence_score_pairs, key=lambda x: x[1])
        
        top_matches[comparison_sentence] = {
            "cv_sentence": top_pair[0],
            "similarity": top_pair[1]
        }
        
        highest_similarities.append(top_pair[1])
    
    # Calculate the average of the highest similarities
    average_similarity = sum(highest_similarities) / len(highest_similarities)

    end_time = time.time()
    processing_time = end_time - start_time

    return jsonify({
        "average_similarity": average_similarity,
        "top_matches": top_matches,
        "processing_time": processing_time
    })

if __name__ == '__main__':
    app.run(debug=True)
