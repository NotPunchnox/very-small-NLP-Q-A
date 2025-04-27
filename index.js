const NeuralNetwork = require('./neural-network');

const nn = new NeuralNetwork();
const trainingData = [
  { inputs: [0, 0], target: 0 },
  { inputs: [0, 1], target: 1 },
  { inputs: [1, 0], target: 1 },
  { inputs: [1, 1], target: 0 }
];

for (let epoch = 0; epoch < 100000; epoch++) {
  let totalLoss = 0;
  for (let data of trainingData) {
    totalLoss += nn.train(data.inputs, data.target);
  }
  const averageLoss = totalLoss / trainingData.length;
  if (epoch % 1000 === 0) {
    console.log(`Epoch ${epoch}, Average Loss: ${averageLoss}`);
  }
}
