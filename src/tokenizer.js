// Training Data (small dataset)
const text = [
  "Bonjour je suis un robot",
  "Un robot est une machine",
  "Il fait beau dehors"
];

class Tokenizer {
  vocab = [];
  pairs = [];

  constructor() { }

  init() {
    // Build vocabulary and pairs
    for (const line of text) {
      const words = line.split(' ');

      // Add words to vocab
      for (const word of words) {
        if (!this.vocab.includes(word)) {
          this.vocab.push(word);
        }
      }

      // Create bigram pairs from words
      for (let i = 0; i < words.length - 1; i++) {
        const pair = [this.vocab.indexOf(words[i]), this.vocab.indexOf(words[i + 1])];
        this.pairs.push(pair);
      }
    }
  }

  tokenize(text) {
    const result = [];
    const text_parsed = text.split(' ');

    //console.log('vocab:', this.vocab);
    console.log('pairs:', this.pairs);

    for (const word of text_parsed) {
      const token = this.vocab.indexOf(word);

      // If token not exist in vocabulary
      if (token === -1) {
        result.push('<unk>');
      } else result.push(token);

    }

    return result;
  }

  decode(tokens) {
    const result = [];

    for (const token of tokens) {
      if (token < 0 || token >= this.vocab.length) {
        console.warn(`Invalid token "${token}". Skipping.`);
        continue;
      }
      result.push(this.vocab[token]);
    }

    return result;
  }
}

module.exports = { Tokenizer };
