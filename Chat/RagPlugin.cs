using System;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.Extensions.AI;

using Qdrant.Client;



public class RAGPlugin(
	QdrantClient qdrantClient, 
	IEmbeddingGenerator<string, Embedding<float>> generator,
	string collectionName)
{
	readonly QdrantClient _qdrantClient = qdrantClient;
	readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingGenerator = generator;
	readonly string _collectionName = collectionName;


	[KernelFunction, Description("Searches the local knowledge base for information about a specific topic.")]
	public async Task<string> SearchKnowledgeBase([Description("The search query")] string query)
	{
		// Vectorize the user's query

		var embedding = await _embeddingGenerator.GenerateAsync(query);

		// Search Qdrant for relevant information from knowledge base

		var searchResults = await _qdrantClient.SearchAsync(_collectionName, embedding.Vector, limit: 5);
		
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