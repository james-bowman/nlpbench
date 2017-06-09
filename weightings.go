package nlpbench

import (
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/james-bowman/sparse"
)

type Transformer interface {
	Fit(mat64.Matrix) Transformer
	Transform(mat mat64.Matrix) (*mat64.Dense, error)
	FitTransform(mat mat64.Matrix) (*mat64.Dense, error)
}

type TfidfTransformer1 struct {
	transform *mat64.Dense
}

func (t *TfidfTransformer1) Fit(mat mat64.Matrix) Transformer {
	m, n := mat.Dims()

	// build a diagonal matrix from array of term weighting values for subsequent
	// multiplication with term document matrics
	t.transform = mat64.NewDense(m, m, nil)

	for i := 0; i < m; i++ {
		df := 0
		for j := 0; j < n; j++ {
			if mat.At(i, j) != 0 {
				df++
			}
		}
		idf := math.Log(float64(1+n) / float64(1+df))
		t.transform.Set(i, i, idf)
	}

	return t
}

func (t *TfidfTransformer1) Transform(mat mat64.Matrix) (*mat64.Dense, error) {
	m, n := mat.Dims()
	product := mat64.NewDense(m, n, nil)

	// simply multiply the matrix by our idf transform (the diagonal matrix of term weights)
	product.Mul(t.transform, mat)

	return product, nil
}

func (t *TfidfTransformer1) FitTransform(mat mat64.Matrix) (*mat64.Dense, error) {
	return t.Fit(mat).Transform(mat)
}

// TfidfTransformer takes a raw term document matrix and weights each raw term frequency
// value depending upon how commonly it occurs across all documents within the corpus.
// For example a very commonly occuring word like `the` is likely to occur in all documents
// and so would be weighted down.
// More precisely, TfidfTransformer applies a tf-idf algorithm to the matrix where each
// term frequency is multiplied by the inverse document frequency.  Inverse document
// frequency is calculated as log(n/df) where df is the number of documents in which the
// term occurs and n is the total number of documents within the corpus.  We add 1 to both n
// and df before division to prevent division by zero.
type TfidfTransformer2 struct {
	weights []float64
}

// NewTfidfTransformer constructs a new TfidfTransformer.
func NewTfidfTransformer() *TfidfTransformer2 {
	return &TfidfTransformer2{}
}

// Fit takes a training term document matrix, counts term occurances across all documents
// and constructs an inverse document frequency transform to apply to matrices in subsequent
// calls to Transform().
func (t *TfidfTransformer2) Fit(mat mat64.Matrix) Transformer {
	m, n := mat.Dims()

	t.weights = make([]float64, m)

	for i := 0; i < m; i++ {
		df := 0
		for j := 0; j < n; j++ {
			if mat.At(i, j) != 0 {
				df++
			}
		}
		idf := math.Log(float64(1+n) / float64(1+df))
		t.weights[i] = idf
	}

	return t
}

func (t *TfidfTransformer2) Transform(mat mat64.Matrix) (*mat64.Dense, error) {
	m, n := mat.Dims()
	product := mat64.NewDense(m, n, nil)

	// iterate over every element of the matrix in turn and
	// multiply the element value by the corresponding term weight
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			product.Set(i, j, mat.At(i, j)*t.weights[i])
		}
	}

	return product, nil
}

// FitTransform is exactly equivalent to calling Fit() followed by Transform() on the
// same matrix.  This is a convenience where separate trianing data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
func (t *TfidfTransformer2) FitTransform(mat mat64.Matrix) (*mat64.Dense, error) {
	return t.Fit(mat).Transform(mat)
}

type TfidfTransformer3 struct {
	weights []float64
}

// Fit takes a training term document matrix, counts term occurances across all documents
// and constructs an inverse document frequency transform to apply to matrices in subsequent
// calls to Transform().
func (t *TfidfTransformer3) Fit(mat mat64.Matrix) Transformer {
	m, n := mat.Dims()

	t.weights = make([]float64, m)

	for i := 0; i < m; i++ {
		df := 0
		for j := 0; j < n; j++ {
			if mat.At(i, j) != 0 {
				df++
			}
		}
		idf := math.Log(float64(1+n) / float64(1+df))
		t.weights[i] = idf
	}

	return t
}

func (t *TfidfTransformer3) Transform(mat mat64.Matrix) (*mat64.Dense, error) {
	m, n := mat.Dims()
	product := mat64.NewDense(m, n, nil)

	// apply a function to every element of the matrix in turn which
	// multiplies the element value by the corresponding term weight
	product.Apply(func(i, j int, v float64) float64 {
		return (v * t.weights[i])
	}, mat)

	return product, nil
}

// FitTransform is exactly equivalent to calling Fit() followed by Transform() on the
// same matrix.  This is a convenience where separate trianing data is not being
// used to fit the model i.e. the model is fitted on the fly to the test data.
func (t *TfidfTransformer3) FitTransform(mat mat64.Matrix) (*mat64.Dense, error) {
	return t.Fit(mat).Transform(mat)
}

type SparseTfidfTransformer struct {
	transform mat64.Matrix
}

func (t *SparseTfidfTransformer) Fit(mat mat64.Matrix) *SparseTfidfTransformer {
	m, n := mat.Dims()

	weights := make([]float64, m)

	csr, ok := mat.(*sparse.CSR)

	for i := 0; i < m; i++ {
		df := 0
		if ok {
			df = csr.RowNNZ(i)
		} else {
			for j := 0; j < n; j++ {
				if mat.At(i, j) != 0 {
					df++
				}
			}
		}
		idf := math.Log(float64(1+n) / float64(1+df))
		weights[i] = idf
	}

	// build a diagonal matrix from array of term weighting values for subsequent
	// multiplication with term document matrics
	t.transform = sparse.NewDIA(m, weights)

	return t
}

func (t *SparseTfidfTransformer) Transform(mat mat64.Matrix) (mat64.Matrix, error) {
	product := &sparse.CSR{}

	// simply multiply the matrix by our idf transform (the diagonal matrix of term weights)
	product.Mul(t.transform, mat)

	return product, nil
}

func (t *SparseTfidfTransformer) FitTransform(mat mat64.Matrix) (mat64.Matrix, error) {
	return t.Fit(mat).Transform(mat)
}
