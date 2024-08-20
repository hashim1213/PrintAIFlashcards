import os
import csv
import pdfplumber
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import time

# Load environment variables from .env file
load_dotenv()

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['CSV_FOLDER'] = 'csv/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['CSV_FOLDER']):
    os.makedirs(app.config['CSV_FOLDER'])

class PDFQAModel:
    def __init__(self, pdf_path, api_key, num_questions):
        self.pdf_path = pdf_path
        self.api_key = api_key
        self.num_questions = num_questions
        self._load_pdf()
        self._create_vectorstore()
        self._create_rag_chain()

    def _load_pdf(self):
        self.docs = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    doc = Document(page_content=text, metadata={"page": i + 1})
                    self.docs.append(doc)

    def _create_vectorstore(self):
        # Create vector store using smaller chunks to generate more questions
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        splits = text_splitter.split_documents(self.docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=self.api_key))
        self.retriever = vectorstore.as_retriever()

    def _create_rag_chain(self):
        system_prompt = (
            "You are an assistant for creating flashcards. "
            "Generate a concise question and answer pair from the provided context. "
            "Format as 'Q: question_text' and 'A: answer_text'."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.api_key)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def generate_flashcards(self):
        flashcards = []
        questions_generated = 0

        while questions_generated < self.num_questions:
            for doc in self.docs:
                if questions_generated >= self.num_questions:
                    break
                question = f"Generate a question and answer from the following text:\n\n{doc.page_content}"
                result = self.rag_chain.invoke({"input": question})
                if "answer" in result:
                    q_and_a = result['answer'].split("A:")
                    if len(q_and_a) == 2:
                        question_text = q_and_a[0].replace("Q:", "").strip()
                        answer_text = q_and_a[1].strip()
                        flashcards.append((question_text, answer_text))
                        questions_generated += 1
                        print(f"Generated flashcard {questions_generated}/{self.num_questions}: {question_text} -> {answer_text}")  # Debugging statement

            # If not enough questions generated, loop through the documents again
            if questions_generated < self.num_questions:
                print("Not enough questions generated, looping through documents again...")

        return flashcards

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/estimate_time', methods=['POST'])
def estimate_time():
    num_questions = int(request.form.get('num_questions', 5))  # Default to 5 questions if not specified
    estimated_time = num_questions * 2  # Rough estimate: 2 seconds per question
    return jsonify({"estimated_time": estimated_time})

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        num_questions = int(request.form.get('num_questions', 5))  # Default to 5 questions if not specified
        print(f"Number of questions requested: {num_questions}")  # Debugging statement
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Simulate processing time for demonstration purposes
            time.sleep(num_questions * 2)  # Simulate the estimated processing time

            # Use the PDFQAModel to generate flashcards
            pdf_qa_model = PDFQAModel(pdf_path=filepath, api_key=os.getenv("OPENAI_API_KEY"), num_questions=num_questions)
            flashcards = pdf_qa_model.generate_flashcards()

            print(f"Number of flashcards generated: {len(flashcards)}")  # Debugging statement

            # Create CSV file
            csv_filepath = create_csv(filepath, flashcards)
            return jsonify({
                "csv_preview": preview_csv(csv_filepath, num_lines=num_questions),  # Pass num_questions to control the preview length
                "csv_download": csv_filepath
            })
    return "Invalid file"


def create_csv(filepath, flashcards):
    csv_filename = secure_filename(os.path.splitext(os.path.basename(filepath))[0] + "_flashcards.csv")
    csv_filepath = os.path.join(app.config['CSV_FOLDER'], csv_filename)

    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer'])
        for question, answer in flashcards:
            writer.writerow([question, answer])

    return csv_filepath

def preview_csv(csv_filepath, num_lines=None):
    preview = []
    with open(csv_filepath, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
        preview.append(header)
        for i, row in enumerate(reader):
            if num_lines is not None and i >= num_lines:
                break
            preview.append(row)
    return preview


@app.route('/download_csv')
def download_csv():
    csv_file = request.args.get('csv_file')
    if csv_file and os.path.exists(csv_file):
        return send_file(csv_file, as_attachment=True)
    return "File not found"

if __name__ == '__main__':
    app.run(debug=True)
