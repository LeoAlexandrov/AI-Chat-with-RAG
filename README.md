# AI Chat with RAG

This is purely experimental project for test and research purposes.

### Short summary
- A simple console chat application that uses Retrieval-Augmented Generation (RAG).
- It combines a vector store ([Qdrant](https://qdrant.tech/)) with embeddings served by an Ollama embedding model and chat completions via Semantic Kernel.
- The chat will call a local RAG plugin to retrieve relevant chunks from indexed `.txt` files and include that context when answering.

### Repository pieces
- `Ingest\Program.cs` — Reads `.txt` files, splits them into overlapping chunks, generates embeddings via `OpenAIClient`, and upserts points with dense and sparse vectors into a Qdrant collection (`local_embeddings` by default).
- `Chat\RagPlugin.cs` — A Kernel plugin exposing `SearchKnowledgeBase(query)` that:
  - Vectorizes the query with the same embedding generator,
  - Builds a Qdrant search request,
  - Performs hybrid search on Qdrant (`QueryAsync`) and returns concatenated chunk contents as context.
- `Chat\Program.cs` — Console chat program:
  - Builds a Semantic Kernel and registers the `RAGPlugin`.
  - Uses `AddOpenAIChatCompletion(modelId, endpoint, apiKey)` configured to call the local Ollama endpoint as the chat LLM.
  - Streams LLM output to the console via `GetStreamingChatMessageContentsAsync`.
- `Common\TextSplitter.cs` — Utility to split documents into `CHUNK_SIZE`/`CHUNK_OVERLAP` chunks used by the ingester.
- `Common\Knowledgebase.cs` — Loads the knowledge base files, splits them into chunks, and provides access to the content.

### Key configuration (edit at top of `Ingest\Program.cs` and `Chat\Program.cs`)
- `KNOWLEDGEBASE_FOLDER` — folder with `.txt .md .html` files to index (Ingest).
- `QDRANT_HOST` / `QDRANT_COLLECTION` — Qdrant host and collection name.
- `OLLAMA_URI` — URL of your Ollama instance (embeddings + optional chat provider).
- `EMBEDDING_MODEL` — name of the embedding model (e.g., `embeddinggemma`).
- `MODEL` — name of the chat model (e.g., `gemma4:26b`).
- `APIKEY` — optional API key.
- `VECTOR_DIMENSIONS`, `CHUNKS_BATCH` — ingestion tuning.

### How it works
1. Indexing: `Ingest` splits each file into overlapping chunks, calls the embedding model in batches and upserts dense and sparse vectors + payload (`content`, `source`) into Qdrant.
2. Chat: `Chat` builds a Semantic Kernel, registers `RAGPlugin`, and runs an interactive loop:
   - User input is added to chat history.
   - Kernel/LLM is allowed to call the registered plugin automatically (per system prompt rules).
   - When invoked, `RAGPlugin.SearchKnowledgeBase` returns the most relevant chunks from Qdrant which the LLM can use as context.
   - Responses are streamed and printed to the console and appended to conversation history.

### Running locally (quickstart)
1. Ensure Qdrant is installed and running.
```
docker pull qdrant/qdrant

docker run --name Qdrant -p 6333:6333 -p 6334:6334 --restart=always \
    -v /usr/share/qdrant/storage:/qdrant/storage \
    -v /usr/share/qdrant/snapshots:/qdrant/snapshots \
    qdrant/qdrant
```
`/usr/share/qdrant` is an example path for persistent storage; adjust as needed for your environment.

2. Ollama is installed and running.  
Here are [download links](https://ollama.com/download) and [quick start guide](https://docs.ollama.com/quickstart).  The project expects an embedding model to be available in Ollama (e.g., `embeddinggemma`) and optionally a chat completion model if you want to use Ollama for that instead of another provider.  
`ollama pull embeddinggemma`  
`ollama pull gemma4:26b`.

2. Edit configuration constants at the top of `Ingest\Program.cs` and `Chat\Program.cs` as needed.
3. Index your KB: from repo root `dotnet run --project Ingest`
4. Start chat: `dotnet run --project Chat`  
Type in the console. Type `exit` to quit.

### Notes & limitations
- `Ingest` currently deletes and recreates the Qdrant collection on each run (data is replaced).
- Minimal error handling and no retry/backoff logic; network/model failures will surface as exceptions.
- `VECTOR_DIMENSIONS` must match the model embedding size (e.g., 768 or 1024).
- The system prompt in `Chat\Program.cs` instructs the assistant to call the RAG plugin when necessary — correctness depends on prompt + kernel tool invocation behavior.
- The chat uses the `AddOpenAIChatCompletion` connector pointed at an Ollama endpoint (or another compatible endpoint) for chat completions.
