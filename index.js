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
