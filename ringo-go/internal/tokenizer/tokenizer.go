// Package tokenizer provides a WordPiece tokeniser compatible with bert-base-uncased.
package tokenizer

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"
)

const (
	tokenUnknown = "[UNK]"
	tokenMask    = "[MASK]"
	tokenPad     = "[PAD]"
	tokenCLS     = "[CLS]"
	tokenSEP     = "[SEP]"
)

// Tokenizer holds the vocabulary and special-token IDs for a bert-base-uncased model.
type Tokenizer struct {
	vocab    map[string]int
	invVocab []string
	MaskID   int
	PadID    int
	CLSID    int
	SEPID    int
}

// Load reads a vocab.txt file (one token per line, line index == token ID) and
// returns an initialised Tokenizer.
func Load(vocabPath string) (*Tokenizer, error) {
	f, err := os.Open(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: open vocab file: %w", err)
	}
	defer f.Close()

	vocab := make(map[string]int)
	var invVocab []string

	scanner := bufio.NewScanner(f)
	id := 0
	for scanner.Scan() {
		token := scanner.Text()
		vocab[token] = id
		invVocab = append(invVocab, token)
		id++
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("tokenizer: read vocab file: %w", err)
	}

	lookup := func(tok string) int {
		if v, ok := vocab[tok]; ok {
			return v
		}
		return 0 // fall back to [PAD] if special token is absent
	}

	t := &Tokenizer{
		vocab:    vocab,
		invVocab: invVocab,
		MaskID:   lookup(tokenMask),
		PadID:    lookup(tokenPad),
		CLSID:    lookup(tokenCLS),
		SEPID:    lookup(tokenSEP),
	}
	return t, nil
}

// Encode converts text to a sequence of token IDs using basic tokenisation
// followed by WordPiece sub-word splitting.
// The special token [MASK] is preserved as-is and never further split.
func (t *Tokenizer) Encode(text string) []int {
	words := t.basicTokenise(text)
	var ids []int
	for _, w := range words {
		ids = append(ids, t.wordPiece(w)...)
	}
	return ids
}

// Decode converts a slice of token IDs back to a human-readable string.
// Sub-word tokens prefixed with "##" are joined directly onto the preceding token.
// When skipSpecial is true, PAD / MASK / CLS / SEP tokens are omitted.
func (t *Tokenizer) Decode(ids []int, skipSpecial bool) string {
	specialIDs := map[int]bool{
		t.PadID:  true,
		t.MaskID: true,
		t.CLSID:  true,
		t.SEPID:  true,
	}

	var sb strings.Builder
	for _, id := range ids {
		if skipSpecial && specialIDs[id] {
			continue
		}
		tok := t.idToToken(id)
		if strings.HasPrefix(tok, "##") {
			sb.WriteString(tok[2:])
		} else {
			if sb.Len() > 0 {
				sb.WriteByte(' ')
			}
			sb.WriteString(tok)
		}
	}
	return sb.String()
}

// DecodeOne converts a single token ID to its string representation,
// stripping any "##" sub-word prefix.
func (t *Tokenizer) DecodeOne(id int) string {
	tok := t.idToToken(id)
	if strings.HasPrefix(tok, "##") {
		return tok[2:]
	}
	return tok
}

// --- internal helpers ---

// basicTokenise lowercases the text and splits it around punctuation and
// whitespace, preserving [MASK] as an atomic unit.
func (t *Tokenizer) basicTokenise(text string) []string {
	text = strings.ToLower(text)

	// Expand the text into rune-level tokens, keeping [mask] intact.
	var words []string
	remaining := text
	for len(remaining) > 0 {
		// Check for [mask] (already lowercased).
		if strings.HasPrefix(remaining, "[mask]") {
			words = append(words, "[MASK]")
			remaining = remaining[len("[mask]"):]
			continue
		}
		r, size := firstRune(remaining)
		remaining = remaining[size:]
		switch {
		case unicode.IsSpace(r):
			// skip whitespace
		case isPunctuation(r):
			words = append(words, string(r))
		default:
			// Accumulate consecutive non-space, non-punctuation runes.
			word := string(r)
			for len(remaining) > 0 {
				if strings.HasPrefix(remaining, "[mask]") {
					break
				}
				nr, ns := firstRune(remaining)
				if unicode.IsSpace(nr) || isPunctuation(nr) {
					break
				}
				word += string(nr)
				remaining = remaining[ns:]
			}
			words = append(words, word)
		}
	}
	return words
}

// wordPiece splits a single word into the longest matching sub-word sequence
// from the vocabulary.  Returns [UNK] if any segment cannot be matched.
func (t *Tokenizer) wordPiece(word string) []int {
	// Special tokens are returned directly without splitting.
	if id, ok := t.vocab[word]; ok && isSpecialToken(word) {
		return []int{id}
	}

	unkID := t.vocab[tokenUnknown]

	var ids []int
	start := 0
	for start < len(word) {
		end := len(word)
		found := false
		for end > start {
			substr := word[start:end]
			if start > 0 {
				substr = "##" + substr
			}
			if id, ok := t.vocab[substr]; ok {
				ids = append(ids, id)
				start = end
				found = true
				break
			}
			end--
		}
		if !found {
			return []int{unkID}
		}
	}
	return ids
}

// idToToken returns the string token for a given ID, or [UNK] if out of range.
func (t *Tokenizer) idToToken(id int) string {
	if id < 0 || id >= len(t.invVocab) {
		return tokenUnknown
	}
	return t.invVocab[id]
}

// firstRune returns the first rune in s together with its byte length.
func firstRune(s string) (rune, int) {
	for _, r := range s {
		return r, len(string(r))
	}
	return 0, 0
}

// isPunctuation reports whether r is an ASCII punctuation character or a
// Unicode symbol / punctuation category member.
func isPunctuation(r rune) bool {
	if r >= 33 && r <= 47 || r >= 58 && r <= 64 || r >= 91 && r <= 96 || r >= 123 && r <= 126 {
		return true
	}
	return unicode.IsPunct(r) || unicode.IsSymbol(r)
}

// isSpecialToken reports whether a token string is a BERT special token.
func isSpecialToken(s string) bool {
	switch s {
	case tokenUnknown, tokenMask, tokenPad, tokenCLS, tokenSEP:
		return true
	}
	return false
}
