package nlpbench

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"testing"

	"github.com/james-bowman/nlp"
)

func Load(newsgroups ...string) []string {
	root := "../datasets/20-newsgroups"
	var files []string

	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			if path == root || len(newsgroups) == 0 {
				return nil
			}
			var found bool
			for _, dir := range newsgroups {
				if root+"/"+dir == path {
					found = true
				}
			}
			if !found {
				return filepath.SkipDir
			}
			return nil
		}

		// process files
		b, err := ioutil.ReadFile(path)
		if err != nil {
			log.Printf("Failed to read file: '%s' caused by %s\n", path, err.Error())
		}
		files = append(files, string(b))

		return nil
	})

	return files
}

// Benchmark stop word removal datastructure/algorithms

// Baseline with no stop word removal
func BenchmarkCountVectoriserFitWithNoStopWordRemoval(b *testing.B) {
	files := Load()

	vect := NewCountVectoriser1(false)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		vect.Fit(files...)
	}
}

// Map based Stop word lookup and removal
func BenchmarkCountVectoriserFitWithMapStopWordRemoval(b *testing.B) {
	files := Load()

	vect := NewCountVectoriser2(true)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		vect.Fit(files...)
	}
}

// Regex (trie) based stop word lookup and removal
func BenchmarkCountVectoriserFitWithRegexStopWordRemoval(b *testing.B) {
	files := Load()

	vect := NewCountVectoriser1(true)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		vect.Fit(files...)
	}
}

// Go implemented Trie based stop word lookup and removal
func BenchmarkCountVectoriserFitWithTrieStopWordRemoval(b *testing.B) {
	files := Load()

	vect := NewCountVectoriser3(true)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		vect.Fit(files...)
	}
}

// Benchmark feature extraction vectorisation into Dense vs Sparse matrices

// Baseline Dense matrix vectorisation
func BenchmarkDenseCountVectoriserTransform(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewCountVectoriser1(false)
	vect.Fit(files...)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		vect.Transform(files...)
	}
}

// Benchmark DOK Sparse matrix vectorisation
func BenchmarkDOKCountVectoriserTransform(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		vect.Transform(files...)
	}
}

func BenchmarkDenseTfidfFitTransform(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	trans := &TfidfTransformer3{}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.FitTransform(mat)
	}
}

func BenchmarkSparseTfidfFitTransform(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	trans := &SparseTfidfTransformer{}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.FitTransform(mat.ToCSR())
	}
}

func BenchmarkDenseApplyTfidfFitWithDense(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	trans := &TfidfTransformer3{}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.Fit(mat)
	}
}

func BenchmarkDenseApplyTfidfFitWithDOK(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	trans := &TfidfTransformer3{}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.Fit(mat)
	}
}

func BenchmarkDenseApplyTfidfFitWithCSR(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	csr := mat.ToCSR()

	trans := &TfidfTransformer3{}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.Fit(csr)
	}
}

func BenchmarkCSRTfidfFitWithCSR(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)
	csr := mat.ToCSR()

	trans := &SparseTfidfTransformer{}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.Fit(csr)
	}
}

func BenchmarkDenseApplyTfidfTransform(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	trans := &TfidfTransformer3{}
	trans.Fit(mat)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.Transform(mat)
	}
}

func BenchmarkDenseApplyTfidfTransformWithDOK(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	trans := &TfidfTransformer3{}
	trans.Fit(mat)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.Transform(mat)
	}
}

func BenchmarkDenseApplyTfidfTransformWithCSR(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)
	csr := mat.ToCSR()

	trans := &TfidfTransformer3{}
	trans.Fit(csr)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.Transform(csr)
	}
}

func BenchmarkDenseApplyTfidfTransformWithConvFromDOKToCSR(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	trans := &TfidfTransformer3{}
	trans.Fit(mat)

	b.ResetTimer()
	csr := mat.ToCSR()
	for n := 0; n < b.N; n++ {
		trans.Transform(csr)
	}
}

/*
func BenchmarkDenseMulTfidfTransform(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	trans := &TfidfTransformer1{}
	trans.Fit(mat)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.Transform(mat)
	}
}

func BenchmarkDenseMulTfidfTransformWithCSR(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)
	csr := mat.ToCSR()

	trans := &TfidfTransformer1{}
	trans.Fit(csr)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.Transform(csr)
	}
}
*/
func BenchmarkCSRTfidfTransformWithDOK(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	trans := &SparseTfidfTransformer{}
	trans.Fit(mat)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.Transform(mat)
	}
}

func BenchmarkCSRTfidfTransformWithCSR(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)
	csr := mat.ToCSR()

	trans := &SparseTfidfTransformer{}
	trans.Fit(csr)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.Transform(csr)
	}
}

func BenchmarkDenseSVD(b *testing.B) {
	files := Load("sci.space")

	vect := NewCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	trans := nlp.NewTruncatedSVD(100)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.FitTransform(mat)
	}
}

func BenchmarkDOKSVD(b *testing.B) {
	files := Load("sci.space")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)

	trans := nlp.NewTruncatedSVD(100)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.FitTransform(mat)
	}
}

func BenchmarkCSRSVD(b *testing.B) {
	files := Load("sci.space")

	vect := NewDOKCountVectoriser1(false)
	vect.Fit(files...)
	mat, _ := vect.Transform(files...)
	csr := mat.ToCSR()

	trans := nlp.NewTruncatedSVD(100)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		trans.FitTransform(csr)
	}
}

func BenchmarkDenseEndToEndVectAndTrans(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	b.ResetTimer()

	vect := NewCountVectoriser1(false)
	trans := &TfidfTransformer3{}

	for n := 0; n < b.N; n++ {
		mat, _ := vect.FitTransform(files...)
		trans.FitTransform(mat)
	}
}

func BenchmarkSparseEndToEndVectAndTrans(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	b.ResetTimer()

	vect := NewDOKCountVectoriser1(false)
	trans := &SparseTfidfTransformer{}

	for n := 0; n < b.N; n++ {
		mat, _ := vect.FitTransform(files...)
		trans.FitTransform(mat.ToCSR())
	}
}

func BenchmarkDenseEndToEndFull(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	b.ResetTimer()

	vect := NewCountVectoriser1(false)
	trans := &TfidfTransformer3{}
	red := nlp.NewTruncatedSVD(100)

	for n := 0; n < b.N; n++ {
		mat, _ := vect.FitTransform(files...)
		tfidf, _ := trans.FitTransform(mat)
		red.FitTransform(tfidf)
	}
}

func BenchmarkSparseEndToEndFull(b *testing.B) {
	files := Load("sci.space", "sci.electronics")

	b.ResetTimer()

	vect := NewDOKCountVectoriser1(false)
	trans := &SparseTfidfTransformer{}
	red := nlp.NewTruncatedSVD(100)

	for n := 0; n < b.N; n++ {
		mat, _ := vect.FitTransform(files...)
		csr := mat.ToCSR()
		tfidf, _ := trans.FitTransform(csr)
		red.FitTransform(tfidf)
	}
}
