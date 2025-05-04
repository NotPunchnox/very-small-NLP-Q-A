// neural-network.js
const math = require('./src/math.js');

class NeuralNetwork {
  constructor(vocabSize, maxSeqLength, embeddingDim = 64, hiddenSize = 128) {
    this.vocabSize = vocabSize;
    this.maxSeqLength = maxSeqLength;
    this.embeddingDim = embeddingDim;
    this.hiddenSize = hiddenSize;

    // Initialisation des poids avec Xavier
    const xavierScale1 = Math.sqrt(2 / (vocabSize + embeddingDim));
    const xavierScale2 = Math.sqrt(2 / (embeddingDim * maxSeqLength + hiddenSize));
    const xavierScale3 = Math.sqrt(2 / (hiddenSize + vocabSize));

    // Embedding : vocabSize -> embeddingDim
    this.embedding = Array(vocabSize).fill().map(() =>
      Array(embeddingDim).fill().map(() => Math.random() * 2 * xavierScale1 - xavierScale1)
    );

    // Couche cachée : (maxSeqLength * embeddingDim) -> hiddenSize
    this.weightsInputHidden = Array(maxSeqLength * embeddingDim).fill().map(() =>
      Array(hiddenSize).fill().map(() => Math.random() * 2 * xavierScale2 - xavierScale2)
    );
    this.biasHidden = [Array(hiddenSize).fill(0)];

    // Couche de sortie : hiddenSize -> vocabSize
    this.weightsHiddenOutput = Array(hiddenSize).fill().map(() =>
      Array(vocabSize).fill().map(() => Math.random() * 2 * xavierScale3 - xavierScale3)
    );
    this.biasOutput = [Array(vocabSize).fill(0)];

    this.learningRate = 0.01;
  }

  // Convertir une séquence de tokens en embeddings
  tokensToEmbedding(tokens) {
    const padded = [...tokens];
    while (padded.length < this.maxSeqLength) {
      // 0 == token <pad>
      padded.push(0);
    }

    if (padded.length > this.maxSeqLength) {
      padded.length = this.maxSeqLength;
    }

    // Convertir chaque token en embedding
    const embedding = padded.map(token => this.embedding[token] || Array(this.embeddingDim).fill(0));
    // Aplatir en un vecteur 1D
    return embedding.flat();
  }

  // Propagation avant : prend une séquence de tokens et retourne une distribution sur le vocabulaire
  forward(tokens) {
    // Convertir les tokens en embeddings
    this.input = [this.tokensToEmbedding(tokens)];

    // Couche cachée
    let hiddenInput = math.matrixAdd(
      math.matrixMultiply(this.input, this.weightsInputHidden),
      this.biasHidden
    );
    this.hidden = hiddenInput.map(row => row.map(math.sigmoid));

    // Couche de sortie (distribution sur le vocabulaire)
    let outputInput = math.matrixAdd(
      math.matrixMultiply(this.hidden, this.weightsHiddenOutput),
      this.biasOutput
    );
    // Appliquer softmax pour obtenir des probabilités
    this.output = outputInput.map(row => {
      const expSum = row.reduce((sum, val) => sum + Math.exp(val), 0);
      return row.map(val => Math.exp(val) / expSum);
    });

    // Distribution sur le vocabulaire
    return this.output[0];
  }

  // Entraînement : prend une question et une réponse tokenisées
  train(questionTokens, answerTokens) {
    let totalLoss = 0;
    const context = [...questionTokens];

    // Générer chaque token de la réponse
    for (let i = 0; i < answerTokens.length; i++) {
      // Propagation avant avec le contexte actuel
      const outputProbs = this.forward(context);

      // Calcul de la perte (entropie croisée)
      const target = Array(this.vocabSize).fill(0);
      target[answerTokens[i]] = 1;
      const loss = math.crossEntropyLoss(outputProbs, target);
      totalLoss += loss;

      // Calcul des gradients
      const outputError = outputProbs.map((prob, j) => prob - target[j]);
      const outputDelta = outputError.map((err, j) =>
        err * math.sigmoidDerivative(outputProbs[j])
      );

      // Erreur de la couche cachée
      const hiddenError = math.matrixMultiply(
        [outputDelta],
        math.matrixTranspose(this.weightsHiddenOutput)
      );
      const hiddenDelta = hiddenError[0].map((val, j) =>
        val * math.sigmoidDerivative(this.hidden[0][j])
      );

      // Mise à jour des poids et biais
      // Couche cachée -> sortie
      const hiddenOutputAdjustment = math.matrixScalarMultiply(
        math.matrixMultiply(math.matrixTranspose(this.hidden), [outputDelta]),
        this.learningRate
      );
      this.weightsHiddenOutput = math.matrixAdd(
        this.weightsHiddenOutput,
        hiddenOutputAdjustment
      );
      this.biasOutput = math.matrixAdd(
        this.biasOutput,
        math.matrixScalarMultiply([outputDelta], this.learningRate)
      );

      // Couche d'entrée -> cachée
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

      // Mise à jour des embeddings
      const inputFlat = this.input[0];
      for (let t = 0; t < Math.min(context.length, this.maxSeqLength); t++) {
        if (context[t] === 0) continue; // Ignorer le padding
        const embeddingGrad = Array(this.embeddingDim).fill(0);
        for (let j = 0; j < this.hiddenSize; j++) {
          for (let k = 0; k < this.embeddingDim; k++) {
            embeddingGrad[k] += hiddenDelta[j] * this.weightsInputHidden[t * this.embeddingDim + k][j];
          }
        }
        for (let k = 0; k < this.embeddingDim; k++) {
          this.embedding[context[t]][k] -= this.learningRate * embeddingGrad[k];
        }
      }

      // Ajouter le token cible au contexte pour la prochaine itération
      context.push(answerTokens[i]);
    }

    return totalLoss / answerTokens.length;
  }

  // Génération : générer une réponse à partir d'une question
  generate(questionTokens, maxLength = 50) {
    const context = [...questionTokens];
    const response = [];

    for (let i = 0; i < maxLength; i++) {
      const outputProbs = this.forward(context);
      // Choisir le token avec la plus haute probabilité
      const nextToken = outputProbs.indexOf(Math.max(...outputProbs));

      // 0 == EOS ( end of sequence )
      if (nextToken === 0) break;
      response.push(nextToken);
      context.push(nextToken);

      if (context.length > this.maxSeqLength) {
        // Maintenir la longueur maximale
        context.shift();
      }
    }

    return response;
  }
}

module.exports = NeuralNetwork;
