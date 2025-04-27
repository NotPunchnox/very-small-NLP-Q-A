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
    this.biasHidden = [[Math.random() * 2 - 1, Math.random() * 2 - 1]],
      this.biasHidden = [[Math.random() * 2 - 1]];

    this.learningRate = 0.1;

  }


  // Fonciton de propagation avant
  forward(inputs) {

    // Couche cachée
    this.input = [inputs];

    let hiddenInput = matrixAdd(
      math.matrixMultiply(this.input, this.weightsInputHidden),
      this.biasHidden
    );
    this.hidden = hiddenInput.map(row => row.map(math.sigmoid));

    let outputInput = matrixAdd(
      matrixMultiply(this.hidden, this.weightsInputHidden),
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




  }
}
