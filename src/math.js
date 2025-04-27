function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
  return x * (1 - x);
}

function matrixMultiply(a, b) {
  const rowsA = a.length;
  const colsA = a[0].length;
  const rowsB = b.length;
  const colsB = b[0].length;
  const result = Array(rowsA).fill().map(() => Array(colsB).fill(0));

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
    return result;
  }
}

function matrixAdd(a, b) {
  console.log("Matrix a:", a, "\nMatrix b:", b)
  return a.map((row, i) => row.map((val, j) => val + b[i][j]));
}

function matrixScalarMultiply(matrix, scalar) {
  return matrix.map(row => row.map(val => val * scalar));
}

function matrixTranspose(matrix) {
  return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}


module.exports = {
  sigmoid,
  sigmoidDerivative,
  matrixMultiply,
  matrixAdd,
  matrixScalarMultiply,
  matrixTranspose
}
