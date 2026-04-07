using System;
using System.Collections.Generic;


public static class TextSplitter
{
	/// <summary>
	/// Splits the input text into chunks of a specified maximum character length, 
	/// with a specified overlap between chunks. The method attempts to split at natural breakpoints
	/// such as newlines, spaces, or punctuation.
	/// </summary>
	/// <param name="text">Text to split</param>
	/// <param name="maxChars">Maximum number of characters per chunk</param>
	/// <param name="overlap">Number of characters to overlap between chunks</param>
	/// <returns>List of text chunks</returns>
	public static List<string> Split(string text, int maxChars, int overlap)
	{
		var chunks = new List<string>();
		int cursor = 0;
		int end = 0;

		while (end < text.Length)
		{
			end = Math.Min(cursor + maxChars, text.Length);

			if (end < text.Length)
			{
				int bp;
				int bpN = text.LastIndexOf('\n', end, end - cursor);
				int bpS = text.LastIndexOf(' ', end, end - cursor);

				if (bpN > bpS)
					bp = bpN;
				else if (bpS > bpN)
					bp = bpS;
				else
				{
					bp = text.LastIndexOf('.', end, end - cursor);

					if (bp < 0)
						bp = text.LastIndexOf(';', end, end - cursor) + 1;

					if (bp < 0)
						bp = text.LastIndexOf(')', end, end - cursor) + 1;

					if (bp < 0)
						bp = text.LastIndexOf('}', end, end - cursor) + 1;

					if (bp < 0)
						bp = text.LastIndexOf(',', end, end - cursor) + 1;

				}

				if (bp > cursor + (maxChars / 2))
					end = bp;
			}

			chunks.Add(text[cursor..end].Trim());
			
			int tCursor = end - overlap;

			while (tCursor > cursor && !char.IsWhiteSpace(text[tCursor])) 
				tCursor--;

			if (tCursor == cursor)
				tCursor = end - overlap;

			cursor = tCursor;
		}

		return chunks;
	}
}