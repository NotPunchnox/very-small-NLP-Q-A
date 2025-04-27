//Importation des fonctions Math
const math = require('./src/math.js');

// Class du réseau de neurones

class NeuralNetwork {
  constructor() {
    // Initilisation des poids aléatoires (-1 à 1)

    this.weightsInputHidden = [
      [Math.random() * 2 - 1, Math.random() * 2 - 1],
      [Math.random() * 2 - 1, Math.random() * 2 - 1]
    ];
    this.weightsHiddenOutput = [
      [Math.random() * 2 - 1],
      [Math.random() * 2 - 1]
    ];

    // Initilisation des biais
    this.biasHidden = [[Math.random() * 2 - 1, Math.random() * 2 - 1]];
    this.biasOutput = [[Math.random() * 2 - 1]];

    this.learningRate = 0.1;

  }


  // Fonction de propagation avant
  forward(inputs) {

    // Couche cachée
    this.input = [inputs];

    let hiddenInput = math.matrixAdd(
      math.matrixMultiply(this.input, this.weightsInputHidden),
      this.biasHidden
    );
    //console.log({ hiddenInput })
    this.hidden = hiddenInput.map(row => row.map(math.sigmoid));

    //console.log("Hidden:", this.hidden, "\nweightsInputHidden:", this.weightsInputHidden);
    //console.log("Multiplication:", math.matrixMultiply(this.hidden, this.weightsInputHidden));

    let outputInput = math.matrixAdd(
      math.matrixMultiply(this.hidden, this.weightsInputHidden),
      this.biasOutput
    );
    this.output = outputInput.map(row => row.map(math.sigmoid));

    return this.output[0][0];
  }


  train(inputs, target) {

    // Propagation avant
    this.forward(inputs);

    // Erreur de sortie
    const outputError = target - this.output[0][0];
    const outputDelta = outputError * math.sigmoidDerivative(this.output[0][0]);

    // Erreur de la couche cachée
    const hiddenError = math.matrixMultiply(
      [[outputDelta]],
      math.matrixTranspose(this.weightsHiddenOutput)
    );
    const hiddenDelta = hiddenError[0].map((val, i) => val * math.sigmoidDerivative(this.hidden[0][i]));

    // Mise à jours des poids et des biais
    // Couche cachée -> sortie
    const hiddenOutputAdjustment = math.matrixScalarMultiply(
      math.matrixMultiply(math.matrixTranspose(this.hidden), [[outputDelta]]),
      this.learningRate
    );
    this.weightsHiddenOutput = math.matrixAdd(
      this.weightsHiddenOutput,
      hiddenOutputAdjustment
    );
    this.biasOutput = math.matrixAdd(
      this.biasOutput,
      math.matrixScalarMultiply([[outputDelta]], this.learningRate)
    );

    // Entrée -> couche cachée
    const inputHiddenAdjustment = math.matrixScalarMultiply(
      math.matrixMultiply(math.matrixTranspose(this.input), [hiddenDelta]),
      this.learningRate
    );
    this.weightsInputHidden = math.matrixAdd(
      this.weightsInputHidden,
      inputHiddenAdjustment
    );
    this.biasHidden = math.matrixAdd(
      this.biasHidden,
      math.matrixScalarMultiply([hiddenDelta], this.learningRate)
    );

  }
}


module.exports = NeuralNetwork;
