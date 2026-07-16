from email.mime import text
from urllib import response

import chromadb
from openai import OpenAI


class AcademicAdvisorRAG:

    def __init__(
        self,
        db_path="./chroma_db",
        collection_name="dsu_aiml_curriculum",
        llm_model="qwen3:1.7b",
        embedding_model="nomic-embed-text"
    ):

        self.embedding_model = embedding_model
        self.llm_model = llm_model

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
    def retrieve_context(self, question, top_k=3):
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