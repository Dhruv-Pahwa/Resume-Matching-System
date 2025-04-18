from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

def create_plot(resume_names, scores):
    plt.figure(figsize=(12, 6))
    colors = []
    for score in scores:
        if score >= 0.5:
            colors.append('green')
        elif score >= 0.3:
            colors.append('orange')
        else:
            colors.append('red')
    
    plt.scatter(resume_names, scores, c=colors, s=100)
    plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
    plt.text(len(resume_names)-1, 0.32, 'Minimum Threshold', color='red')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Similarity Score (0-1)')
    plt.title('Resume Matching Scores')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return plot_data

def generate_feedback(score):
    if score >= 0.5:
        return ("Excellent match! This resume strongly aligns with the job requirements.", 'success')
    elif score >= 0.3:
        return ("Moderate match. This resume has some relevant qualifications but may need additional screening.", 'warning')
    else:
        return ("Not suitable for this position. The candidate lacks required qualifications based on this resume.", 'danger')

@app.route("/")
def matchresume():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        if not resume_files or not job_description:
            return render_template('matchresume.html', error="Please upload resumes and enter a job description.")

        resumes = []
        valid_files = []
        for resume_file in resume_files:
            try:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
                resume_file.save(filename)
                text = extract_text(filename)
                if text.strip():
                    resumes.append(text)
                    valid_files.append(resume_file.filename)
            except Exception as e:
                print(f"Error processing {resume_file.filename}: {str(e)}")

        if not resumes:
            return render_template('matchresume.html', error="Could not extract text from any resumes. Please try different files.")

        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()
        similarities = cosine_similarity([vectors[0]], vectors[1:])[0]

        scored_resumes = sorted(zip(valid_files, similarities), key=lambda x: x[1], reverse=True)
        
        results = []
        for filename, score in scored_resumes:
            feedback, feedback_class = generate_feedback(score)
            results.append({
                'filename': filename,
                'score': round(score, 2),
                'feedback': feedback,
                'feedback_class': feedback_class
            })

        plot_data = create_plot([r['filename'] for r in results], [r['score'] for r in results])

        return render_template('matchresume.html', results=results, plot_url=plot_data)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
