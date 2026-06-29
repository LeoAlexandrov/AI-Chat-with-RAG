using System;
using System.Buffers;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;
using Qdrant.Client;
using Qdrant.Client.Grpc;


/// <summary>
/// This experimental plugin searches relevant chunks in Qdrant database like the basic plugin. 
/// Each vector has additional payload data (added by ingestor) with chunk offset and length in original knowledge file.
/// Using this data the plugin extends the chunk by substracting (BLOCK_SIZE - {chunkLength}) / 4 from the offset
/// and setting length to BLOCK_SIZE.
/// The context built by this plugin is larger than the context built by the basic plugin. 
/// This plugin requires original knowledge files.
/// </summary>
public class RagPlugin(
	QdrantClient qdrantClient,
	IEmbeddingGenerator<string, Embedding<float>> generator,
	KnowledgeBase knowledgeBase,
	string collectionName)
{
	const int BLOCK_SIZE = 4096;

	readonly QdrantClient _qdrantClient = qdrantClient;
	readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingGenerator = generator;
	readonly KnowledgeBase _knowledgeBase = knowledgeBase;
	readonly string _collectionName = collectionName;

	public event EventHandler<string> OnContextReady = null;


	struct Section : IComparable<Section>
	{
		public long Offset;
		public int Length;
		public string Source;
		public int Order;

		public readonly int CompareSource(Section other) =>
			string.Compare(
				Source,
				other.Source,
				Environment.OSVersion.Platform == PlatformID.Win32NT ?
					StringComparison.InvariantCultureIgnoreCase :
					StringComparison.InvariantCulture
			);

		public readonly int CompareTo(Section other)
		{
			int r = CompareSource(other);

			return r == 0 ? Offset.CompareTo(other.Offset) : r;
		}
	}

	struct SectionReadResult : IComparable<SectionReadResult>
	{
		public int Index;
		public int Order;
		public byte[] Data;

		public readonly int CompareTo(SectionReadResult other) => Index.CompareTo(other.Index);
	}

	class SectionOrderComparer : IComparer<SectionReadResult>
	{
		public int Compare(SectionReadResult x, SectionReadResult y) => x.Order.CompareTo(y.Order);
	}

	static List<Section> JoinSections(List<Section> sections)
	{
		int n = sections.Count;
		var result = new List<Section>(n);

		Section lastAdded = default;
		int i = 0;

		while (i < n)
		{
			if (i == 0 || lastAdded.CompareSource(sections[i]) != 0 || sections[i].Offset > lastAdded.Offset + lastAdded.Length)
			{
				lastAdded = sections[i];
				result.Add(lastAdded);
			}
			else
			{
				long l = sections[i].Offset - lastAdded.Offset + sections[i].Length;

				if (l <= int.MaxValue)
				{
					lastAdded.Length = (int)l;
					result[^1] = lastAdded;
				}
				else
				{
					lastAdded = sections[i];
					result.Add(lastAdded);
				}
			}

			i++;
		}

		return result;
	}

	static async Task<SectionReadResult[]> ReadSectionsAsync(string path, ReadOnlyMemory<Section> sections)
	{
		using var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read);
		var tasks = new Task<SectionReadResult>[sections.Length];
		var span = sections.Span;

		for (int i = 0; i < span.Length; i++)
		{
			int idx = i;
			var (offset, length, order) = (span[i].Offset, span[i].Length, span[i].Order);

			tasks[i] = Task.Run(async () =>
			{
				byte[] buffer = GC.AllocateUninitializedArray<byte>(length);

				int read = await RandomAccess.ReadAsync(handle, buffer, offset);

				if (read != length)
					throw new IOException("Unexpected EOF");

				return new SectionReadResult() { Index = idx, Data = buffer };
			});
		}

		var result = await Task.WhenAll(tasks);

		result.Sort();

		return result;
	}

	static async Task<SectionReadResult[]> ReadSectionsAsync(Section[] sections)
	{
		int n = sections.Length;
		int pos = 0;
		var result = new SectionReadResult[n];

		while (pos < n)
		{
			string kbFile = sections[pos].Source;
			int len = sections.Skip(pos).TakeWhile(s => s.Source == kbFile).Count();
			var mem = new ReadOnlyMemory<Section>(sections, pos, len);
			var res = await ReadSectionsAsync(kbFile, mem);

			Array.Copy(res, 0, result, pos, res.Length);
			pos += len;
		}

		Array.Sort(result, new SectionOrderComparer());

		return result;
	}

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


		if (searchResults.Count != 0)
		{
			var sections = new List<Section>(searchResults.Count);
			int ord = 0;

			foreach (var sr in searchResults)
			{
				string kbFile = sr.Payload["source"].StringValue;
				long maxLen = _knowledgeBase.GetFileSize(kbFile);

				if (maxLen == 0)
					continue;

				long offs = sr.Payload["offset"].IntegerValue;
				int len = (int)sr.Payload["length"].IntegerValue;

				offs -= (BLOCK_SIZE - len) / 4;

				if (offs < 0)
					offs = 0;

				len = BLOCK_SIZE;

				if (offs + len > maxLen)
					len = (int)(maxLen - offs);

				sections.Add(new Section() { Offset = offs, Length = len, Source = kbFile, Order = ord++ });
			}

			sections.Sort();
			var toRead = JoinSections(sections);
			var res = await ReadSectionsAsync([.. toRead]);

			var extraContext = res.Select(r => Encoding.UTF8.GetString(r.Data));

			result = "Relevant information from the knowledge base:\n" + string.Join("\n---\n", extraContext);
		}
		else
		{
			result = "No relevant information found in the knowledge base.";
		}

#if DEBUG
		System.Diagnostics.Debug.WriteLine(result);
#endif

		RaiseOnContextReadyEvent(result);

		return result;
	}

	void RaiseOnContextReadyEvent(string context)
	{
		EventHandler<string> handler = OnContextReady;
		handler?.Invoke(this, context);
	}

}