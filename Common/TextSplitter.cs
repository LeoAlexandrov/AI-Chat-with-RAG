using System;
using System.Collections.Generic;


public static class TextSplitter
{
	/// <summary>
	/// Splits the utf-8 byte array into chunks of a specified maximum character length, 
	/// with a specified overlap between chunks. The method attempts to split at natural breakpoints
	/// such as newlines, spaces, or punctuation.
	/// </summary>
	/// <param name="utf8Text">Utf-8 text to split</param>
	/// <param name="maxChars">Maximum number of characters per chunk</param>
	/// <param name="overlap">Number of characters to overlap between chunks</param>
	/// <returns>List of Ranges in utf-8 byte array representing text chunk</returns>

	public static List<Range> Split(byte[] utf8Text, int maxChars, int overlap)
	{
		var chunks = new List<Range>();
		int cursor = 0;
		int nextCursor;
		int end = 0;

		while (end < utf8Text.Length)
		{
			nextCursor = Math.Min(cursor + maxChars, utf8Text.Length);

			if (nextCursor < utf8Text.Length)
			{
				end = nextCursor - 1;

				while (end > cursor && utf8Text[end] > 127) 
					end--;

				if (end == cursor)
				{
					end = nextCursor;
					continue;
				}

				while (end > cursor && utf8Text[end] != (byte)('\n') && utf8Text[end] != (byte)(' '))
					end--;

				if (end == cursor)
				{
					end = nextCursor - 1;

					while (end > cursor
						&& utf8Text[end] != (byte)('.')
						&& utf8Text[end] != (byte)(';')
						&& utf8Text[end] != (byte)(')')
						&& utf8Text[end] != (byte)('}')
						&& utf8Text[end] != (byte)(',')) end--;
				}

				if (end == cursor || end < (nextCursor - cursor) / 2)
					end = nextCursor - 1;
			}

			chunks.Add(new Range(cursor, end));

			int tCursor = end - overlap;

			while (tCursor > cursor && (utf8Text[tCursor] > 127 ||!char.IsWhiteSpace((char)utf8Text[tCursor])))
				tCursor--;

			if (tCursor == cursor)
				tCursor = end - overlap;

			cursor = tCursor;
			end = nextCursor;
		}

		return chunks;

	}


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