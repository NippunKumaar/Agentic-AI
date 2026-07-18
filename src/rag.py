from email.mime import text
from urllib import response

from annotated_types import doc
import chromadb
from click import prompt
from openai import OpenAI
from src.parser import PDFParser
import re
def normalize(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower())

class AcademicAdvisorRAG:

    def __init__(
        self,
        db_path="./chroma_db",
        collection_name="dsu_aiml_curriculum",
        llm_model="qwen3:1.7b",
        embedding_model="nomic-embed-text", 
        verbose=True
    ):

        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.verbose = verbose

        self.db = chromadb.PersistentClient(path=db_path)

        self.collection = self.db.get_or_create_collection(
            name=collection_name
        )

        self.llm = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
    
    def generate_embedding(self, text):
        """
        Generate an embedding for the given text using Ollama.
        """
        response = self.llm.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def ingest_documents(self, pdf_path):
        """
        Read syllabus PDF, extract courses and store them in ChromaDB.
        """
        parser = PDFParser(pdf_path)

        text = parser.extract_text()
        lines = parser.split_into_lines(text)
        courses = parser.extract_raw_courses(lines)

        print(f"Found {len(courses)} courses.")

        for i, course in enumerate(courses):

         #   print(course.course_name, "->", course.course_code)

            # Skip malformed entries
            if course.course_code.strip() == "":
                print(f"Skipping malformed course: {course.course_name}")
                continue

            document = f"""
        Course Name: {course.course_name}
        Course Code: {course.course_code}

        {course.raw_text}
        """

            embedding = self.generate_embedding(document)

            self.collection.add(
                ids=[f"course_{i}"],
                documents=[document],
                embeddings=[embedding],
                metadatas=[{
                    "course_code": course.course_code,
                    "course_name": course.course_name
                }]
            )
        print("Knowledge base successfully created.")

    def reset_collection(self):
        """
        Delete and recreate the collection.
        """
        self.db.delete_collection("dsu_aiml_curriculum")

        self.collection = self.db.get_or_create_collection(
            name="dsu_aiml_curriculum"
        )
        print("Collection reset successfully.")
    def retrieve_documents(self, question, top_k=5):
        """
        Retrieve the most relevant curriculum chunks
        for the given question.
        """
        question_lower = question.lower()

        all_docs = self.collection.get(include=["documents", "metadatas"])

        for i, meta in enumerate(all_docs["metadatas"]):
            course_name = meta["course_name"].lower()

            if course_name in question_lower:
                print("✓ Exact course match found:", meta["course_name"])

                return {
                    "documents": [[all_docs["documents"][i]]],
                    "metadatas": [[meta]],
                    "distances": [[0.0]]
                }
        query_embedding = self.generate_embedding(question)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        for i in range(len(results["documents"][0])):
            print("=" * 80)
            print("Distance :", results["distances"][0][i])
            print("Course   :", results["metadatas"][0][i]["course_name"])
            print("Code     :", results["metadatas"][0][i]["course_code"])
        
        for i, doc in enumerate(results["documents"][0]):
            print("=" * 80)
            print(results["metadatas"][0][i]["course_name"])
            print(doc[:500])
        return results
    
    def build_prompt(self, question, results):
        """
        Construct the RAG prompt using retrieved documents.
        """

        documents = results["documents"][0]

        context = "\n\n".join(documents)

        prompt = f"""
            You are an AI Academic Advisor for the Dayananda Sagar University AI & ML curriculum.

            Answer the user's question ONLY using the retrieved curriculum below.

            IMPORTANT RULES:
            1. Treat the first retrieved document as the primary source of truth.
            2. Use information from other retrieved documents ONLY if the user explicitly asks for a comparison or if the first document does not contain the required information.
            3. Do NOT use your own knowledge. if any other questions asked that is not related to curriculam, dont answer it using your knowledge, just say you are an academic advisor and question is out of scope. 
            4. Do NOT define concepts from memory.
            5. Do NOT add information that is not explicitly present in the retrieved curriculum.
            6. If the answer cannot be found in the retrieved curriculum, reply exactly:
            "I could not find this information in the curriculum."

            Retrieved Curriculum:
            {context}

            Question:
            {question}

            Answer:
            """
        return prompt
    def generate_answer(self, prompt):
        """
        Generate the final answer using the LLM.
        """

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content
    def ask(self, question):
        """
        Complete RAG pipeline.
        """

        if(self.verbose):
            print("🔍 Retrieving relevant curriculum documents...")

        retrival_results = self.retrieve_documents(question)
        num_docs = len(retrival_results["documents"][0])


        if(self.verbose):
           print(f"✓ Retrieved {num_docs} relevant document(s).")
           print("\nRetrieved Documents:")
           for i, doc in enumerate(retrival_results["documents"][0]):
                print("="*80)
                print(doc[:300])

        if(self.verbose):
            print("📝 Building prompt..")

        prompt = self.build_prompt(
            question,
            retrival_results
        )

        if(self.verbose):
            print("🤖 Generating response...")

        answer = self.generate_answer(prompt)

        if(self.verbose):
            print("✓ Response generated.")

        return answer