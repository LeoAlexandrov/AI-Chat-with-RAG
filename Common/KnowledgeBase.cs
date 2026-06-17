using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;


public class KnowledgeBase(string kbFolder, string[] patterns)
{
	const float BM25_K1 = 1.5f;
	const float BM25_B = 0.75f;
	const int CHUNK_SIZE = 1024;    // Characters per chunk
	const int CHUNK_OVERLAP = 256;  // Overlap to keep context across boundaries

	readonly Dictionary<string, int> Vocab = [];
	readonly Dictionary<int, float> Idf = [];
	readonly List<Chunk> Chunks = [];
	readonly string KBFolder = kbFolder;
	readonly string[] Patterns = patterns ?? ["*.txt", "*.md", "*.html"];

	float AvgDL;

	public IReadOnlyList<Chunk> GetChunks() => Chunks;

	public struct Chunk
	{
		public string Content { get; set; }
		public string SourceFile { get; set; }
	}

	public struct BM25Encoding
	{
		public uint[] Indices { get; set; }
		public float[] Values { get; set; }
	}

	public async Task Load()
	{
		var kbfiles = Patterns.SelectMany(p => Directory.GetFiles(KBFolder, p, SearchOption.AllDirectories)).ToArray();

		kbfiles.Sort();


		// Read all knowledge files, split them to chunks

		Chunks.Clear();

		foreach (var kbfile in kbfiles)
		{
			var text = await File.ReadAllTextAsync(kbfile);
			Chunks.AddRange(TextSplitter.Split(text, CHUNK_SIZE, CHUNK_OVERLAP).Select(c => new Chunk() { Content = c, SourceFile = kbfile }));
		}


		// Build vocabulary and compute IDF weights

		Vocab.Clear();
		Idf.Clear();

		var allTokens = Chunks.Select(c => Tokenize(c.Content).Distinct().ToArray()).ToArray();

		int N = Chunks.Count;
		AvgDL = allTokens.Average(t => (float)t.Length);


		// Build vocab

		int idx = 0;

		foreach (var token in allTokens.SelectMany(t => t).Distinct())
			Vocab.TryAdd(token, idx++);


		// Compute IDF per term

		var df = new Dictionary<int, int>();

		foreach (var tokens in allTokens)
			foreach (var token in tokens)
				if (Vocab.TryGetValue(token, out int id))
					df[id] = df.GetValueOrDefault(id) + 1;

		foreach (var (id, freq) in df)
			Idf[id] = MathF.Log((N - freq + 0.5f) / (freq + 0.5f) + 1.0f);
	}

	static string[] Tokenize(string text) =>
		text.ToLowerInvariant().Split([' ', '\t', '\n', '\r', '.', ',', '!', '?'], StringSplitOptions.RemoveEmptyEntries);

	public BM25Encoding BM25Encode(string text)
	{
		var tokens = Tokenize(text);

		var tf = tokens
			.GroupBy(t => t)
			.Where(g => Vocab.ContainsKey(g.Key))
			.ToDictionary(g => Vocab[g.Key], g => (float)g.Count());

		var result = tf.Select(kv =>
		{
			float score = Idf.GetValueOrDefault(kv.Key) *
				(kv.Value * (BM25_K1 + 1.0f)) /
				(kv.Value + BM25_K1 * (1.0f - BM25_B + BM25_B * tokens.Length / AvgDL));

			return (Index: (uint)kv.Key, Value: (float)score);
		})
			.Where(x => x.Value > 0)
			.OrderBy(x => x.Index)
			.ToArray();

		return new BM25Encoding
		{
			Indices = result.Select(x => x.Index).ToArray(),
			Values = result.Select(x => x.Value).ToArray()
		};
	}

}
