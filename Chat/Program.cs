using System;
using System.Text;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

using OllamaSharp;
using Qdrant.Client;



// setup console for multilanguage support

Console.InputEncoding = Encoding.UTF8;
Console.OutputEncoding = Encoding.UTF8;


// *** configuration ***

const string OLLAMA_URI = "http://localhost:11434/";
const string QDRANT_HOST = "minipc.local"; // "localhost" or IP/name of the machine running Qdrant
const string QDRANT_COLLECTION = "local_embeddings";
const string EMBEDDING_MODEL = "embeddinggemma"; // or "mxbai-embed-large" 


var ollamaUri = new Uri(OLLAMA_URI);

// ollama provider

string modelId = "gemma4:26b"; // "gpt-oss:120b-cloud", // "qwen3.5:35b"
string apiKey = null; // or real key if you'd like to use cloud LLMs like 'gpt-oss:120b-cloud'
var endpoint = new Uri(ollamaUri, "v1");

/*
// alternative provider cerebras.ai

string modelId = "qwen-3-235b-a22b-instruct-2507";
string apiKey = "<your cerebras.ai api key>";
var endpoint = new Uri("https://api.cerebras.ai/v1");
*/

// Initialize RAG plugin

var qdrantClient = new QdrantClient(QDRANT_HOST);
var embeddingGenerator = new OllamaApiClient(ollamaUri, EMBEDDING_MODEL);
var ragPlugin = new RAGPlugin(qdrantClient, embeddingGenerator, QDRANT_COLLECTION);

// Configure and build Semantic Kernel

var builder = Kernel.CreateBuilder();

builder.Services.AddOpenAIChatCompletion(modelId, endpoint, apiKey);
builder.Plugins.AddFromObject(ragPlugin); 

var kernel = builder.Build();


// *** Start chat ***

Console.WriteLine("Chat with the AI. Type 'exit' to stop.");

var history = new ChatHistory(
@"You are a Q&A assistant. You have access to a RAG plugin that can retrieve relevant information from an external knowledge base.
When answering the user:
1. First, check whether the conversation history already contains enough information to answer the question.
2. If the history does NOT contain the necessary information, you MUST call the RAG plugin to retrieve relevant context before answering.
3. After retrieving context, use both the retrieved chunks and the conversation history to produce a complete and accurate answer.
4. Chunks may overlap or be slightly out of order; combine them into a coherent narrative.
5. If the user explicitly asks you to search, retrieve, or look something up, ALWAYS call the RAG plugin.",
	AuthorRole.System);


OpenAIPromptExecutionSettings execSettings = new() { ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions };

var chat = kernel.GetRequiredService<IChatCompletionService>();
var response = new StringBuilder();

// chat loop

while (true)
{
	Console.Write("\r\nUser > ");
	var input = Console.ReadLine();

	if (string.IsNullOrWhiteSpace(input))
		continue;

	if (input.Trim().Equals("exit", StringComparison.CurrentCultureIgnoreCase)) 
		break;

	history.AddUserMessage(input);
	response.Clear();

	Console.Write($"\r\nAI > thinking...");
	var (cX, cY) = Console.GetCursorPosition();
	Console.ForegroundColor = ConsoleColor.Green;

	try
	{
		var asyncMessage = chat.GetStreamingChatMessageContentsAsync(history, execSettings, kernel);

		await foreach (var msgUpdate in asyncMessage)
		{
			if (response.Length == 0)
				Console.SetCursorPosition(cX - "thinking...".Length, cY);

			Console.Write(msgUpdate.Content);
			response.AppendLine(msgUpdate.Content);
		}

		Console.WriteLine();

		if (response.Length != 0)
		{
			var result = new ChatMessageContent()
			{
				Role = AuthorRole.Assistant,
				Content = response.ToString()
			};

			history.AddMessage(result.Role, result.Content);
		}
		else
		{
			Console.ForegroundColor = ConsoleColor.Yellow;
			Console.WriteLine("No response generated.");
		}

	}
	catch (Exception ex)
	{
		Console.ForegroundColor = ConsoleColor.Red;
		Console.WriteLine($"\r\n{ex.Message}");
	}
	finally
	{
		Console.ForegroundColor = ConsoleColor.White;
	}

	Console.ForegroundColor = ConsoleColor.White;
}