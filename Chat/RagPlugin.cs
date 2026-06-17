using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;
using Qdrant.Client;
using Qdrant.Client.Grpc;



public class RAGPlugin(
	QdrantClient qdrantClient, 
	IEmbeddingGenerator<string, Embedding<float>> generator,
	KnowledgeBase knowledgeBase,
	string collectionName)
{
	readonly QdrantClient _qdrantClient = qdrantClient;
	readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingGenerator = generator;
	readonly KnowledgeBase _knowledgeBase = knowledgeBase;
	readonly string _collectionName = collectionName;


	[KernelFunction, Description("Searches the local knowledge base for information about a specific topic.")]
	public async Task<string> SearchKnowledgeBase([Description("The search query")] string query)
	{
		// Vectorize the user's query

		var embedding = await _embeddingGenerator.GenerateAsync(query);

		// Sparse vector for the query using the same BM25 encoder used at index time

		var sparse = _knowledgeBase.BM25Encode(query);

		// Build (value, index) tuple array that the C# SDK expects for sparse prefetch

		var sparseQuery = sparse.Indices
			.Zip(sparse.Values, (idx, val) => (val, idx))
			.ToArray();

		// Hybrid search: prefetch top-10 from each vector space, then fuse with RRF

		var searchResults = await _qdrantClient.QueryAsync(
			collectionName: _collectionName,
			prefetch: [
				new()
				{
					Query  = embedding.Vector.ToArray(), // float[]  -> dense prefetch
				    Using  = "dense",
					Limit  = 10
				},
				new()
				{
					Query  = sparseQuery,                // (float, uint)[] -> sparse prefetch
				    Using  = "sparse",
					Limit  = 10
				}
			],
			query: Fusion.Rrf,                           // fuse the two ranked lists
			limit: 5
		);

		string result;


		if (searchResults.Count == 0)
		{
			result = "No relevant information found in the local database.";
			System.Diagnostics.Debug.WriteLine(result);
		}
		else
		{
			System.Diagnostics.Debug.WriteLine("Context found:");
			foreach (var sr in searchResults)
				System.Diagnostics.Debug.WriteLine(sr);
			System.Diagnostics.Debug.WriteLine("--------------");

			var extraContext = searchResults.Select(r => r.Payload["content"].ToString());
			result = "Relevant information from local files:\n" + string.Join("\n---\n", extraContext);
		}

		return result;
	}
}