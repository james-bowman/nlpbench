package nlpbench

import (
	"regexp"
	"strings"

	"github.com/golang-collections/collections/trie"
	"github.com/gonum/matrix/mat64"
	"github.com/james-bowman/sparse"
)

var (
	stopWords = []string{"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"}
)

type CountVectoriser1 struct {
	Vocabulary    map[string]int
	wordTokeniser *regexp.Regexp
	stopWords     *regexp.Regexp
}

func NewCountVectoriser1(removeStopwords bool) *CountVectoriser1 {
	var stop *regexp.Regexp

	if removeStopwords {
		reStr := "\\A("

		for i, word := range stopWords {
			if i != 0 {
				reStr += `|`
			}
			reStr += `\Q` + word + `\E`
		}
		reStr += ")\\z"
		stop = regexp.MustCompile(reStr)
	}
	return &CountVectoriser1{
		Vocabulary:    make(map[string]int),
		wordTokeniser: regexp.MustCompile("\\w+"),
		stopWords:     stop,
	}
}

func (v *CountVectoriser1) Fit(train ...string) *CountVectoriser1 {
	i := 0
	for _, doc := range train {
		words := v.tokenise(doc)

		for _, word := range words {
			_, exists := v.Vocabulary[word]
			if !exists {
				// if enabled, remove stop words
				if v.stopWords != nil {
					if v.stopWords.MatchString(word) {
						continue
					}
				}
				v.Vocabulary[word] = i
				i++
			}
		}
	}

	return v
}

func (v *CountVectoriser1) Transform(docs ...string) (*mat64.Dense, error) {
	mat := mat64.NewDense(len(v.Vocabulary), len(docs), nil)

	for d, doc := range docs {
		words := v.tokenise(doc)

		for _, word := range words {
			i, exists := v.Vocabulary[word]

			if exists {
				mat.Set(i, d, mat.At(i, d)+1)
			}
		}
	}
	return mat, nil
}

func (v *CountVectoriser1) FitTransform(docs ...string) (*mat64.Dense, error) {
	return v.Fit(docs...).Transform(docs...)
}

func (v *CountVectoriser1) tokenise(text string) []string {
	// convert content to lower case
	c := strings.ToLower(text)

	// match whole words, removing any punctuation/whitespace
	words := v.wordTokeniser.FindAllString(c, -1)

	return words
}

type CountVectoriser2 struct {
	Vocabulary    map[string]int
	wordTokeniser *regexp.Regexp
	stopWords     map[string]bool
}

func NewCountVectoriser2(removeStopwords bool) *CountVectoriser2 {
	var stop map[string]bool

	if removeStopwords {
		stop = make(map[string]bool)
		for _, word := range stopWords {
			stop[word] = true
		}
	}
	return &CountVectoriser2{
		Vocabulary:    make(map[string]int),
		wordTokeniser: regexp.MustCompile("\\w+"),
		stopWords:     stop,
	}
}

func (v *CountVectoriser2) Fit(train ...string) *CountVectoriser2 {
	i := 0
	for _, doc := range train {
		words := v.tokenise(doc)

		for _, word := range words {
			_, exists := v.Vocabulary[word]
			if !exists {
				// if enabled, remove stop words
				if v.stopWords != nil {
					if v.stopWords[word] {
						continue
					}
				}
				v.Vocabulary[word] = i
				i++
			}
		}
	}

	return v
}

func (v *CountVectoriser2) Transform(docs ...string) (*mat64.Dense, error) {
	mat := mat64.NewDense(len(v.Vocabulary), len(docs), nil)

	for d, doc := range docs {
		words := v.tokenise(doc)

		for _, word := range words {
			i, exists := v.Vocabulary[word]

			if exists {
				mat.Set(i, d, mat.At(i, d)+1)
			}
		}
	}
	return mat, nil
}

func (v *CountVectoriser2) FitTransform(docs ...string) (*mat64.Dense, error) {
	return v.Fit(docs...).Transform(docs...)
}

func (v *CountVectoriser2) tokenise(text string) []string {
	// convert content to lower case
	c := strings.ToLower(text)

	// match whole words, removing any punctuation/whitespace
	words := v.wordTokeniser.FindAllString(c, -1)

	return words
}

type CountVectoriser3 struct {
	Vocabulary    map[string]int
	wordTokeniser *regexp.Regexp
	stopWords     *trie.Trie
}

func NewCountVectoriser3(removeStopwords bool) *CountVectoriser3 {
	var stop *trie.Trie

	if removeStopwords {
		stop = trie.New()
		for _, word := range stopWords {
			stop.Insert(word, true)
		}
	}
	return &CountVectoriser3{
		Vocabulary:    make(map[string]int),
		wordTokeniser: regexp.MustCompile("\\w+"),
		stopWords:     stop,
	}
}

func (v *CountVectoriser3) Fit(train ...string) *CountVectoriser3 {
	i := 0
	for _, doc := range train {
		words := v.tokenise(doc)

		for _, word := range words {
			_, exists := v.Vocabulary[word]
			if !exists {
				// if enabled, remove stop words
				if v.stopWords != nil {
					if v.stopWords.Has(word) {
						continue
					}
				}
				v.Vocabulary[word] = i
				i++
			}
		}
	}

	return v
}

func (v *CountVectoriser3) Transform(docs ...string) (*mat64.Dense, error) {
	mat := mat64.NewDense(len(v.Vocabulary), len(docs), nil)

	for d, doc := range docs {
		words := v.tokenise(doc)

		for _, word := range words {
			i, exists := v.Vocabulary[word]

			if exists {
				mat.Set(i, d, mat.At(i, d)+1)
			}
		}
	}
	return mat, nil
}

func (v *CountVectoriser3) FitTransform(docs ...string) (*mat64.Dense, error) {
	return v.Fit(docs...).Transform(docs...)
}

func (v *CountVectoriser3) tokenise(text string) []string {
	// convert content to lower case
	c := strings.ToLower(text)

	// match whole words, removing any punctuation/whitespace
	words := v.wordTokeniser.FindAllString(c, -1)

	return words
}

type DOKCountVectoriser1 struct {
	Vocabulary    map[string]int
	wordTokeniser *regexp.Regexp
	stopWords     *regexp.Regexp
}

func NewDOKCountVectoriser1(removeStopwords bool) *DOKCountVectoriser1 {
	var stop *regexp.Regexp

	if removeStopwords {
		reStr := "\\A("

		for i, word := range stopWords {
			if i != 0 {
				reStr += `|`
			}
			reStr += `\Q` + word + `\E`
		}
		reStr += ")\\z"
		stop = regexp.MustCompile(reStr)
	}
	return &DOKCountVectoriser1{
		Vocabulary:    make(map[string]int),
		wordTokeniser: regexp.MustCompile("\\w+"),
		stopWords:     stop,
	}
}

func (v *DOKCountVectoriser1) Fit(train ...string) *DOKCountVectoriser1 {
	i := 0
	for _, doc := range train {
		words := v.tokenise(doc)

		for _, word := range words {
			_, exists := v.Vocabulary[word]
			if !exists {
				// if enabled, remove stop words
				if v.stopWords != nil {
					if v.stopWords.MatchString(word) {
						continue
					}
				}
				v.Vocabulary[word] = i
				i++
			}
		}
	}

	return v
}

func (v *DOKCountVectoriser1) Transform(docs ...string) (*sparse.DOK, error) {
	mat := sparse.NewDOK(len(v.Vocabulary), len(docs))

	for d, doc := range docs {
		words := v.tokenise(doc)

		for _, word := range words {
			i, exists := v.Vocabulary[word]

			if exists {
				mat.Set(i, d, mat.At(i, d)+1)
			}
		}
	}
	return mat, nil
}

func (v *DOKCountVectoriser1) FitTransform(docs ...string) (*sparse.DOK, error) {
	return v.Fit(docs...).Transform(docs...)
}

func (v *DOKCountVectoriser1) tokenise(text string) []string {
	// convert content to lower case
	c := strings.ToLower(text)

	// match whole words, removing any punctuation/whitespace
	words := v.wordTokeniser.FindAllString(c, -1)

	return words
}
