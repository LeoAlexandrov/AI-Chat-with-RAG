using System;
using System.ClientModel;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

using OpenAI;
using Qdrant.Client;



// setup console for multilanguage support

Console.InputEncoding = Encoding.UTF8;
Console.OutputEncoding = Encoding.UTF8;


// *** configuration ***

const string KNOWLEDGEBASE_FOLDER = @"C:\Temp\Kb";
const string QDRANT_HOST = "minipc.local";             // "localhost" or IP/name of the machine running Qdrant
const string QDRANT_COLLECTION = "local_embeddings";

const string OLLAMA_URI = "http://localhost:11434/v1"; // http://localhost:8080/v1 for llama.cpp
const string EMBEDDING_MODEL = "embeddinggemma";       // "EmbeddingGemma"; //"embeddinggemma-300M-BF16"; // "mxbai-embed-large" 
const string MODEL = "gemma4:26b";                     // "Gemma4-26b-q4"; // "google_gemma-4-26B-A4B-it-Q4_K_M"; //"google_gemma-4-E4B-it-Q4_K_M"; // "gpt-oss:120b-cloud", // "qwen3.5:35b"
const string APIKEY = "0";


// ollama provider

var endpoint = new Uri(OLLAMA_URI);


// Load all knowledge base files

var kb = new KnowledgeBase(KNOWLEDGEBASE_FOLDER, ["*.txt", "*.md", "*.html"]);
await kb.Load();


// Create embedding generator

var clientOptions = new OpenAIClientOptions { Endpoint = endpoint };
var openAIClient = new OpenAIClient(new ApiKeyCredential(APIKEY), clientOptions);

var embeddingGenerator = openAIClient
	.GetEmbeddingClient(EMBEDDING_MODEL)
	.AsIEmbeddingGenerator();


// Initialize RAG plugin

var qdrantClient = new QdrantClient(QDRANT_HOST);
var ragPlugin = new RAGPlugin(qdrantClient, embeddingGenerator, kb, QDRANT_COLLECTION);


// Configure and build Semantic Kernel

var builder = Kernel.CreateBuilder();

builder.Services.AddOpenAIChatCompletion(MODEL, endpoint, APIKEY);
builder.Plugins.AddFromObject(ragPlugin); 

var kernel = builder.Build();


// *** Start chat ***

Console.WriteLine("Chat with the AI. Type 'exit' to stop.");

var history = new ChatHistory(
@"You are useful assistant. You have access to a RAG plugin that can retrieve relevant information from an external knowledge base.
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

	input += @"\nIf you need to retrieve information from the local knowledge base to answer this question, use the RAG plugin.
If no relevant information is found, respond according your knowledges, but mark the answer.";
//Do not answer the question without using the plugin if you determine that relevant information is not already present in the conversation history.";

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