from email.mime import text
from urllib import response

import chromadb
from click import prompt
from openai import OpenAI


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
    
    def ingest_documents(self, file_path):
        """
        Read a text file, split it into chunks,
        generate embeddings and store them in ChromaDB.
        """

        # Read file
        with open(file_path, "r", encoding="utf-8") as file:
            document = file.read()

        # Simple semantic chunking
        chunks = document.split("---------------------------------------------------------")
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        print(f"Found {len(chunks)} chunks.")

        # Store every chunk
        for i, chunk in enumerate(chunks):

            embedding = self.generate_embedding(chunk)

            self.collection.add(
                ids=[f"course_{i}"],
                documents=[chunk],
                embeddings=[embedding]
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
    def retrieve_documents(self, question, top_k=3):
        """
        Retrieve the most relevant curriculum chunks
        for the given question.
        """

        query_embedding = self.generate_embedding(question)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results
    def build_prompt(self, question, results):
        """
        Construct the RAG prompt using retrieved documents.
        """
        documents = results["documents"][0]

        context = "\n\n".join(documents)

        prompt = f"""
    You are the DSU AI Academic Advisor.

    Your responsibility is to answer student questions ONLY using the retrieved curriculum information.

    If the answer is not available in the retrieved documents, reply exactly:

    "I could not find this information in the curriculum."

    Retrieved Curriculum:
    ---------------------
    {context}

    Student Question:
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
           for i, doc in enumerate(retrival_results["documents"][0], 1):
               title = doc.split("\n")[0]
               print(f"{i}. {title}")

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