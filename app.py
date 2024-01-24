from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

@app.route('/compare-resume', methods=['POST'])
def compare_resume():
    # Get the resume file from the request
    resume_file = request.files['resume']

    # Open the resume file using PyPDF2
    pdf_reader = PyPDF2.PdfReader(resume_file)
    resume_page = pdf_reader.pages[-1]
    resume_text = resume_page.extract_text()

    # Load the job description file
    job_description_file = request.files['job_desc']
    job_description_pdf_reader = PyPDF2.PdfReader(job_description_file)
    job_description_page = job_description_pdf_reader.pages[-1]
    job_description_text = job_description_page.extract_text()

    # Perform similarity comparison
    text = [resume_text, job_description_text]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text)
    similarity_scores = cosine_similarity(count_matrix)

    # Calculate the match percentage
    match_percentage = similarity_scores[0][1] * 100
    match_percentage = round(match_percentage, 2)

    # Prepare the response
    response = {
        'match_percentage': match_percentage
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()

