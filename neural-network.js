const math = require('./src/math.js');

class NeuralNetwork {
  constructor() {
    // Architecture : 2 entrées, 2 couches cachées (4 neurones chacune), 1 sortie
    const inputSize = 2;
    const hiddenSize1 = 4;
    const hiddenSize2 = 4;
    const outputSize = 1;

    // Initialisation des poids avec l'initialisation Xavier
    const xavierScale1 = Math.sqrt(2 / (inputSize + hiddenSize1));
    const xavierScale2 = Math.sqrt(2 / (hiddenSize1 + hiddenSize2));
    const xavierScale3 = Math.sqrt(2 / (hiddenSize2 + outputSize));

    this.weightsInputHidden1 = Array(inputSize).fill().map(() =>
      Array(hiddenSize1).fill().map(() => Math.random() * 2 * xavierScale1 - xavierScale1)
    );
    this.weightsHidden1Hidden2 = Array(hiddenSize1).fill().map(() =>
      Array(hiddenSize2).fill().map(() => Math.random() * 2 * xavierScale2 - xavierScale2)
    );
    this.weightsHidden2Output = Array(hiddenSize2).fill().map(() =>
      Array(outputSize).fill().map(() => Math.random() * 2 * xavierScale3 - xavierScale3)
    );

    // Initialisation des biais à 0 pour simplifier
    this.biasHidden1 = [Array(hiddenSize1).fill(0)];
    this.biasHidden2 = [Array(hiddenSize2).fill(0)];
    this.biasOutput = [Array(outputSize).fill(0)];

    this.learningRate = 0.1; // Taux d'apprentissage augmenté
  }

  forward(inputs) {
    // Couche d'entrée vers première couche cachée
    this.input = [inputs];
    let hiddenInput1 = math.matrixAdd(
      math.matrixMultiply(this.input, this.weightsInputHidden1),
      this.biasHidden1
    );
    this.hidden1 = hiddenInput1.map(row => row.map(math.sigmoid));

    // Première couche cachée vers deuxième couche cachée
    let hiddenInput2 = math.matrixAdd(
      math.matrixMultiply(this.hidden1, this.weightsHidden1Hidden2),
      this.biasHidden2
    );
    this.hidden2 = hiddenInput2.map(row => row.map(math.sigmoid));

    // Deuxième couche cachée vers sortie
    let outputInput = math.matrixAdd(
      math.matrixMultiply(this.hidden2, this.weightsHidden2Output),
      this.biasOutput
    );
    this.output = outputInput.map(row => row.map(math.sigmoid));

    return this.output[0][0];
  }

  train(inputs, target) {
    // Propagation avant
    this.forward(inputs);

    // Calcul de la perte (erreur quadratique moyenne)
    const loss = 0.5 * Math.pow(target - this.output[0][0], 2);

    // Erreur de sortie
    const outputError = target - this.output[0][0];
    const outputDelta = outputError * math.sigmoidDerivative(this.output[0][0]);

    // Erreur de la deuxième couche cachée
    const hiddenError2 = math.matrixMultiply(
      [[outputDelta]],
      math.matrixTranspose(this.weightsHidden2Output)
    );
    const hiddenDelta2 = hiddenError2[0].map((val, i) =>
      val * math.sigmoidDerivative(this.hidden2[0][i])
    );

    // Erreur de la première couche cachée
    const hiddenError1 = math.matrixMultiply(
      [hiddenDelta2],
      math.matrixTranspose(this.weightsHidden1Hidden2)
    );
    const hiddenDelta1 = hiddenError1[0].map((val, i) =>
      val * math.sigmoidDerivative(this.hidden1[0][i])
    );

    // Mise à jour des poids et biais
    // Deuxième couche cachée -> sortie
    const hidden2OutputAdjustment = math.matrixScalarMultiply(
      math.matrixMultiply(math.matrixTranspose(this.hidden2), [[outputDelta]]),
      this.learningRate
    );
    this.weightsHidden2Output = math.matrixAdd(
      this.weightsHidden2Output,
      hidden2OutputAdjustment
    );
    this.biasOutput = math.matrixAdd(
      this.biasOutput,
      math.matrixScalarMultiply([[outputDelta]], this.learningRate)
    );

    // Première couche cachée -> deuxième couche cachée
    const hidden1Hidden2Adjustment = math.matrixScalarMultiply(
      math.matrixMultiply(math.matrixTranspose(this.hidden1), [hiddenDelta2]),
      this.learningRate
    );
    this.weightsHidden1Hidden2 = math.matrixAdd(
      this.weightsHidden1Hidden2,
      hidden1Hidden2Adjustment
    );
    this.biasHidden2 = math.matrixAdd(
      this.biasHidden2,
      math.matrixScalarMultiply([hiddenDelta2], this.learningRate)
    );

    // Entrée -> première couche cachée
    const inputHidden1Adjustment = math.matrixScalarMultiply(
      math.matrixMultiply(math.matrixTranspose(this.input), [hiddenDelta1]),
      this.learningRate
    );
    this.weightsInputHidden1 = math.matrixAdd(
      this.weightsInputHidden1,
      inputHidden1Adjustment
    );
    this.biasHidden1 = math.matrixAdd(
      this.biasHidden1,
      math.matrixScalarMultiply([hiddenDelta1], this.learningRate)
    );

    return loss;
  }
}

module.exports = NeuralNetwork;
