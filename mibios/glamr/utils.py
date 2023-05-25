import re


query_word_pat = re.compile(r"""("[^"]*"|'[^']*'|\S+)\s*""")


def split_query(query, keep_quotes=False):
    """
    Split a search query into word-like things, respecting quotes
    """
    words = []
    while query:
        match = query_word_pat.match(query)
        if not match:
            break
        word = match.group(1)  # match w/o trailing space
        query = query[match.end():]

        if not keep_quotes:
            match word[0]:
                case '"':
                    word = word.strip('"')
                case "'":
                    word = word.strip("'")
        if word:
            words.append(word)
    return words
