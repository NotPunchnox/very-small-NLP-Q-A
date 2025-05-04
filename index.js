const NeuralNetwork = require('./neural-network');
const Tokenizer = require('./src/tokenizer.js');

// Créer et initialiser le tokenizer
const tokenizer = new Tokenizer();
tokenizer.init();

// Créer et initialiser le réseau neuronal
const nn = new NeuralNetwork();
const trainingData = [
  { inputs: [1, 2], target: 0.3 },
  { inputs: [2, 3], target: 0.4 },
  { inputs: [4, 5], target: 0.6 },
  { inputs: [6, 7], target: 0.8 }
];

// Entrainer le réseau de neurones
for (let epoch = 0; epoch < 1000000; epoch++) {
  let totalLoss = 0;
  for (let data of trainingData) {
    totalLoss += nn.train(data.inputs, data.target);
  }
  const averageLoss = totalLoss / trainingData.length;
  if (epoch % 1000 === 0) {
    console.log(`Epoch ${epoch}, Average Loss: ${averageLoss}`);
  }
}



// Inférence/test du réseau neuronal après l'entrainement
for (const data of trainingData) {
  let output = nn.forward(data.inputs);
  console.log(`Inputs: ${data.inputs} | result: ${output} | Attendu: ${data.target}`);
}
