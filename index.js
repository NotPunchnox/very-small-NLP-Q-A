// index.js
const NeuralNetwork = require('./neural-network');

// Supposons que vous avez une fonction tokenizer
const tokenizer = {
  encode: (text) => {
    // Exemple simplifié : convertir le texte en tokens
    return text.toLowerCase().split(' ').map(word => word.charCodeAt(0) % 100);
  },
  decode: (tokens) => {
    // Exemple simplifié : convertir les tokens en texte
    return tokens.map(token => String.fromCharCode(token + 32)).join('');
  },
  vocabSize: 100 // Taille du vocabulaire (à ajuster selon votre tokenizer)
};

// Exemple de données Q&A
const trainingData = [
  {
    question: "What is the capital of France?",
    answer: "The capital of France is Paris."
  },
  {
    question: "Who wrote Romeo and Juliet?",
    answer: "Romeo and Juliet was written by William Shakespeare."
  }
];

// Convertir les données en tokens
const tokenizedData = trainingData.map(({ question, answer }) => ({
  questionTokens: tokenizer.encode(question),
  answerTokens: tokenizer.encode(answer)
}));

// Créer le réseau
const maxSeqLength = Math.max(
  ...tokenizedData.map(data => data.questionTokens.length + data.answerTokens.length)
);
const nn = new NeuralNetwork(tokenizer.vocabSize, maxSeqLength);

// Entraînement
const epochs = 1000;
for (let i = 0; i < epochs; i++) {
  let totalLoss = 0;
  for (const data of tokenizedData) {
    totalLoss += nn.train(data.questionTokens, data.answerTokens);
  }
  if (i % 100 === 0) {
    console.log(`Epoch ${i}, Average Loss: ${(totalLoss / tokenizedData.length).toFixed(6)}`);
  }
}

// Test de génération
console.log("Test Q&A:");
for (const { question } of trainingData) {
  const questionTokens = tokenizer.encode(question);
  const responseTokens = nn.generate(questionTokens);
  const response = tokenizer.decode(responseTokens);
  console.log(`Question: ${question}`);
  console.log(`Réponse: ${response}`);
}
