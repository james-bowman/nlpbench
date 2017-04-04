package nlpbench

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func benchmarkFit(t Transformer, m, n int, b *testing.B) {
	mat := mat64.NewDense(m, n, nil)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		t.Fit(mat)
	}
}

func benchmarkFitTransform(t Transformer, m, n int, b *testing.B) {
	mat := mat64.NewDense(m, n, nil)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		t.FitTransform(mat)
	}
}

func benchmarkTransform(t Transformer, m, n int, b *testing.B) {
	mat := mat64.NewDense(m, n, nil)
	t.Fit(mat)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		t.Transform(mat)
	}
}

func BenchmarkTFIDF1Fit30x3(b *testing.B) {
	benchmarkFit(&TfidfTransformer1{}, 30, 3, b)
}
func BenchmarkTFIDF1Fit300x30(b *testing.B) {
	benchmarkFit(&TfidfTransformer1{}, 300, 30, b)
}
func BenchmarkTFIDF1Fit3000x300(b *testing.B) {
	benchmarkFit(&TfidfTransformer1{}, 3000, 300, b)
}
func BenchmarkTFIDF1Fit30000x3000(b *testing.B) {
	benchmarkFit(&TfidfTransformer1{}, 30000, 3000, b)
}

func BenchmarkTFIDF1Transform30x3(b *testing.B) {
	benchmarkTransform(&TfidfTransformer1{}, 30, 3, b)
}
func BenchmarkTFIDF1Transform300x30(b *testing.B) {
	benchmarkTransform(&TfidfTransformer1{}, 300, 30, b)
}
func BenchmarkTFIDF1Transform3000x300(b *testing.B) {
	benchmarkTransform(&TfidfTransformer1{}, 3000, 300, b)
}
func BenchmarkTFIDF1Transform30000x3000(b *testing.B) {
	benchmarkTransform(&TfidfTransformer1{}, 30000, 3000, b)
}

func BenchmarkTFIDF1FitTransform30x3(b *testing.B) {
	benchmarkFitTransform(&TfidfTransformer1{}, 30, 3, b)
}
func BenchmarkTFIDF1FitTransform300x30(b *testing.B) {
	benchmarkFitTransform(&TfidfTransformer1{}, 300, 30, b)
}
func BenchmarkTFIDF1FitTransform3000x300(b *testing.B) {
	benchmarkFitTransform(&TfidfTransformer1{}, 3000, 300, b)
}
func BenchmarkTFIDF1FitTransform30000x3000(b *testing.B) {
	benchmarkFitTransform(&TfidfTransformer1{}, 30000, 3000, b)
}

// TFIDF 2
func BenchmarkTFIDF2Fit30x3(b *testing.B) {
	benchmarkFit(&TfidfTransformer2{}, 30, 3, b)
}
func BenchmarkTFIDF2Fit300x30(b *testing.B) {
	benchmarkFit(&TfidfTransformer2{}, 300, 30, b)
}
func BenchmarkTFIDF2Fit3000x300(b *testing.B) {
	benchmarkFit(&TfidfTransformer2{}, 3000, 300, b)
}
func BenchmarkTFIDF2Fit30000x3000(b *testing.B) {
	benchmarkFit(&TfidfTransformer2{}, 30000, 3000, b)
}

func BenchmarkTFIDF2Transform30x3(b *testing.B) {
	benchmarkTransform(&TfidfTransformer2{}, 30, 3, b)
}
func BenchmarkTFIDF2Transform300x30(b *testing.B) {
	benchmarkTransform(&TfidfTransformer2{}, 300, 30, b)
}
func BenchmarkTFIDF2Transform3000x300(b *testing.B) {
	benchmarkTransform(&TfidfTransformer2{}, 3000, 300, b)
}
func BenchmarkTFIDF2Transform30000x3000(b *testing.B) {
	benchmarkTransform(&TfidfTransformer2{}, 30000, 3000, b)
}

func BenchmarkTFIDF2FitTransform30x3(b *testing.B) {
	benchmarkFitTransform(&TfidfTransformer2{}, 30, 3, b)
}
func BenchmarkTFIDF2FitTransform300x30(b *testing.B) {
	benchmarkFitTransform(&TfidfTransformer2{}, 300, 30, b)
}
func BenchmarkTFIDF2FitTransform3000x300(b *testing.B) {
	benchmarkFitTransform(&TfidfTransformer2{}, 3000, 300, b)
}
func BenchmarkTFIDF2FitTransform30000x3000(b *testing.B) {
	benchmarkFitTransform(&TfidfTransformer2{}, 30000, 3000, b)
}

// TFIDF 3
func BenchmarkTFIDF3Transform30x3(b *testing.B) {
	benchmarkTransform(&TfidfTransformer3{}, 30, 3, b)
}
func BenchmarkTFIDF3Transform300x30(b *testing.B) {
	benchmarkTransform(&TfidfTransformer3{}, 300, 30, b)
}
func BenchmarkTFIDF3Transform3000x300(b *testing.B) {
	benchmarkTransform(&TfidfTransformer3{}, 3000, 300, b)
}
func BenchmarkTFIDF3Transform30000x3000(b *testing.B) {
	benchmarkTransform(&TfidfTransformer3{}, 30000, 3000, b)
}

func BenchmarkTFIDF3FitTransform30000x3000(b *testing.B) {
	benchmarkFitTransform(&TfidfTransformer3{}, 30000, 3000, b)
}
