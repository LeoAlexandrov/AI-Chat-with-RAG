using System;
using System.ClientModel;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.AI;

using OpenAI;
using Qdrant.Client;
using Qdrant.Client.Grpc;



// Configuration

const string KNOWLEDGEBASE_FOLDER = @"C:\Temp\Kb";
const string QDRANT_HOST = "minipc.local";             // "localhost" or IP/name of the machine running Qdrant
const string QDRANT_COLLECTION = "local_embeddings";

const string OLLAMA_URI = "http://localhost:11434/v1"; // http://localhost:8080/v1 for llama.cpp
const string EMBEDDING_MODEL = "embeddinggemma";       // "EmbeddingGemma"; // "embeddinggemma-300M-BF16"; //"mxbai-embed-large"
const string APIKEY = "0";

const int VECTOR_DIMENSIONS = 768;                     // 768 for embeddinggemma, 1024 for mxbai-embed-large
const int CHUNKS_BATCH = 4;                            // Number of chunks to process in parallel when generating embeddings


// Initialize Qdrant Client and collection

var qdrant = new QdrantClient(QDRANT_HOST);

var collections = await qdrant.ListCollectionsAsync();

if (collections.Contains(QDRANT_COLLECTION))
	await qdrant.DeleteCollectionAsync(QDRANT_COLLECTION);

await qdrant.CreateCollectionAsync(
	QDRANT_COLLECTION,
	vectorsConfig: new VectorParamsMap()
	{
		Map = { ["dense"] = new VectorParams() { Size = VECTOR_DIMENSIONS, Distance = Distance.Cosine } }
	},
	sparseVectorsConfig: ("sparse", new SparseVectorParams())
);


// Initialize embedding generator

var clientOptions = new OpenAIClientOptions { Endpoint = new Uri(OLLAMA_URI) };
var openAIClient = new OpenAIClient(new ApiKeyCredential(APIKEY), clientOptions);
var embeddingGenerator = openAIClient.GetEmbeddingClient(EMBEDDING_MODEL).AsIEmbeddingGenerator();


// Load all knowledge base files

var kb = new KnowledgeBase(KNOWLEDGEBASE_FOLDER, ["*.txt", "*.md", "*.html"]);
await kb.Load();


// Embed all chunks in Qdrant

var chunks = kb.GetChunks();

for (int i = 0; i < chunks.Count; i += CHUNKS_BATCH)
{
	var batch = chunks.Skip(i).Take(CHUNKS_BATCH);
	var embeddings = await embeddingGenerator.GenerateAsync(batch.Select(c => c.Content));
	var sparseVectors = batch.Select(chunk => kb.BM25Encode(chunk.Content)).ToArray();

	var points = embeddings.Select((e, k) => new PointStruct()
			{
				Id = (ulong)(i + k),
				Vectors = new Dictionary<string, Vector>()
				{
					["dense"] = e.Vector.ToArray(),
					["sparse"] = (sparseVectors[k].Values, sparseVectors[k].Indices)
				},
				Payload = { ["content"] = chunks[i + k].Content, ["source"] = chunks[i + k].SourceFile }
			})
		.ToArray();

	await qdrant.UpsertAsync(QDRANT_COLLECTION, points);

	Console.Write(".");
}


Console.WriteLine("\nDone! Your knowledge base is now searchable in Qdrant.");