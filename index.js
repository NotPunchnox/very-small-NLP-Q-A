// Fonction d'activation sigmoïde
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// Dérivée de la sigmoïde
function sigmoidDerivative(x) {
  return x * (1 - x);
}

// Multiplication de matrices
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
  }
  return result;
}

// Addition de matrices
function matrixAdd(a, b) {
  return a.map((row, i) => row.map((val, j) => val + b[i][j]));
}

// Multiplication d'une matrice par un scalaire
function matrixScalarMultiply(matrix, scalar) {
  return matrix.map(row => row.map(val => val * scalar));
}

// Transposition d'une matrice
function matrixTranspose(matrix) {
  return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

// Réseau de neurones
class NeuralNetwork {
  constructor() {
    // Initialisation des poids aléatoires (-1 à 1)
    this.weightsInputHidden = [
      [Math.random() * 2 - 1, Math.random() * 2 - 1],
      [Math.random() * 2 - 1, Math.random() * 2 - 1]
    ];
    this.weightsHiddenOutput = [
      [Math.random() * 2 - 1],
      [Math.random() * 2 - 1]
    ];

    // Initialisation des biais
    this.biasHidden = [[Math.random() * 2 - 1, Math.random() * 2 - 1]];
    this.biasOutput = [[Math.random() * 2 - 1]];

    // Taux d'apprentissage
    this.learningRate = 0.1;
  }

  // Propagation avant
  forward(inputs) {
    // Couche cachée
    this.input = [inputs];
    let hiddenInput = matrixAdd(
      matrixMultiply(this.input, this.weightsInputHidden),
      this.biasHidden
    );
    this.hidden = hiddenInput.map(row => row.map(sigmoid));

    // Couche de sortie
    let outputInput = matrixAdd(
      matrixMultiply(this.hidden, this.weightsHiddenOutput),
      this.biasOutput
    );
    this.output = outputInput.map(row => row.map(sigmoid));

    return this.output[0][0];
  }

  // Entraînement
  train(inputs, target) {
    // Propagation avant
    this.forward(inputs);

    // Erreur de sortie
    const outputError = target - this.output[0][0];
    const outputDelta = outputError * sigmoidDerivative(this.output[0][0]);

    // Erreur de la couche cachée
    const hiddenError = matrixMultiply(
      [[outputDelta]],
      matrixTranspose(this.weightsHiddenOutput)
    );
    const hiddenDelta = hiddenError[0].map((val, i) =>
      val * sigmoidDerivative(this.hidden[0][i])
    );

    // Mise à jour des poids et biais
    // Couche cachée -> sortie
    const hiddenOutputAdjustment = matrixScalarMultiply(
      matrixMultiply(matrixTranspose(this.hidden), [[outputDelta]]),
      this.learningRate
    );
    this.weightsHiddenOutput = matrixAdd(
      this.weightsHiddenOutput,
      hiddenOutputAdjustment
    );
    this.biasOutput = matrixAdd(
      this.biasOutput,
      matrixScalarMultiply([[outputDelta]], this.learningRate)
    );

    // Entrée -> couche cachée
    const inputHiddenAdjustment = matrixScalarMultiply(
      matrixMultiply(matrixTranspose(this.input), [hiddenDelta]),
      this.learningRate
    );
    this.weightsInputHidden = matrixAdd(
      this.weightsInputHidden,
      inputHiddenAdjustment
    );
    this.biasHidden = matrixAdd(
      this.biasHidden,
      matrixScalarMultiply([hiddenDelta], this.learningRate)
    );
  }
}

// Données d'entraînement XOR
const trainingData = [
  { input: [0, 0], target: 0 },
  { input: [0, 1], target: 1 },
  { input: [1, 0], target: 1 },
  { input: [1, 1], target: 0 }
];

// Création et entraînement du réseau
const nn = new NeuralNetwork();
const epochs = 10000;

for (let i = 0; i < epochs; i++) {
  for (const data of trainingData) {
    nn.train(data.input, data.target);
  }
}

// Test du réseau
console.log("Test XOR:");
for (const data of trainingData) {
  const output = nn.forward(data.input);
  console.log(
    `Entrée: [${data.input}], Sortie: ${output.toFixed(4)}, Attendu: ${data.target}`
  );
}
