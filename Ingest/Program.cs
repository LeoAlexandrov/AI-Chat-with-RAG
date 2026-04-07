using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.AI;

using OllamaSharp;
using Qdrant.Client;
using Qdrant.Client.Grpc;



// *** configuration ***

const string KNOWLEDGEBASE_FOLDER = @"C:\Temp\Kb";
const string OLLAMA_URI = "http://localhost:11434/";
const string QDRANT_HOST = "minipc.local"; // "localhost" or IP/name of the machine running Qdrant
const string QDRANT_COLLECTION = "local_embeddings";
const string EMBEDDING_MODEL = "embeddinggemma"; // or "mxbai-embed-large" 

const int VECTOR_DIMENSIONS = 768; // 768 for embeddinggemma, 1024 for mxbai-embed-large
const int CHUNK_SIZE = 1024; // Characters per chunk
const int CHUNK_OVERLAP = 256; // Overlap to keep context across boundaries
const int CHUNKS_BATCH = 4; // Number of chunks to process in parallel when generating embeddings

var ollamaUri = new Uri(OLLAMA_URI);


// Initialize Qdrant Client and collection

var qdrant = new QdrantClient(QDRANT_HOST);

var collections = await qdrant.ListCollectionsAsync();

if (collections.Contains(QDRANT_COLLECTION))
	await qdrant.DeleteCollectionAsync(QDRANT_COLLECTION);

await qdrant.CreateCollectionAsync(
	QDRANT_COLLECTION,
	new VectorParams() { Size = VECTOR_DIMENSIONS, Distance = Distance.Cosine });


// Initialize embedding generator

IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator = new OllamaApiClient(ollamaUri, EMBEDDING_MODEL);


// Get list all knowledge base files

var kbfiles = Directory.GetFiles(KNOWLEDGEBASE_FOLDER, "*.txt", SearchOption.AllDirectories);
int n = kbfiles.Length;
int current = 0;
int i = 0;

// embed each file in Qdrant

foreach (var kbfile in kbfiles)
{
	Console.Write($"Processing file {++current}/{n}: {kbfile} ");

	var text = await File.ReadAllTextAsync(kbfile);
	var chunks = TextSplitter.Split(text, CHUNK_SIZE, CHUNK_OVERLAP);

	for (int j = 0; j < chunks.Count; j += CHUNKS_BATCH)
	{
		var embeddings = await embeddingGenerator.GenerateAsync(chunks.Skip(j).Take(CHUNKS_BATCH));

		var points = embeddings.Select((e, k) => new PointStruct()
			{
				Id = (ulong)(i + j + k),
				Vectors = e.Vector.ToArray(),
				Payload = { ["content"] = chunks[j + k], ["source"] = kbfile }
			}).ToArray();

		await qdrant.UpsertAsync(QDRANT_COLLECTION, points);

		Console.Write(".");
	}

	i+= chunks.Count;
	Console.WriteLine(" done.");
}


Console.WriteLine("\nDone! Your knowledge base is now searchable in Qdrant.");