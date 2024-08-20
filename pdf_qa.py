import pdfplumber
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document  # Correct import

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
                if questions_generated >= self.num_questions:
                    break

            if questions_generated < self.num_questions:
                # Repeat processing of the same documents if more questions are needed
                continue

        return flashcards
