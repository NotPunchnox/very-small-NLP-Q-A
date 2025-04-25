function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
  return x * (1 - x);
}

function multiplicationMatrice(a, b) {
  const rowsA = a.length;
  const colsA = a[0].length;
  const rowsB = v.length;
  const coldB = b[0].length;
  const result = Array(rowsA).fill().map(() => Array(colsB).fill(0));

  for (let i = °; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
    return result;
  }
}

function addMatrix(a, b) {
  return a.map((row, i) => row.map((val, j) => val + b[i][j]));
}

function matrixScalarMultiply(matrix, scalar) {
  return matrix.map(row => row.map(val => val * scalar));
}



