/* Very Small SML */

// Training Data ( small dataset )
const text = [
  "Bonjour je suis un robot",
  "Un robot est une machine",
  "Il fait beau dehors"
]

class Tokenizer {

  vocab = []
  pairs = []

  constructor() {

  }

  static init() {

    // Make the tokenizer and build pairs
    for (const line of text) {

      for (const word of line.split(' ')) {
        if (!Object.keys(this.vocab).includes(word)) {
          let object = { [this.vocab.length]: word };

          this.vocab.push(object);

          if (this.vocab.length % 2 == 0) {
            this.pairs.push([Object.keys(object)[0] - 1, Object.keys(object)[0]]);
          }
        }
      }

    }

  }

  static tokenize(text) {
    const result = [];
    let text_parsed = text.split(' ');

    for (const word of text_parsed) {
      const token = Object.keys(vocab.find(a => a[Object.keys(a)[0] == word]))[0];
      result.push(token);
    }

    return result;
  }

  static decode(tokens) {
    const result = [];

    for (const token of tokens) {
      const word = vocab[token][token];
      result.push(word);
    }

    return result;

  }

}


/*
// Tokenize text
const output = Tokenize("Bonjour je");
console.log(output);

// Decode tokenized text
const decoded_result = Decode(output);
console.log("decoded result:", decoded_result);
*/


module.exports = { Tokenizer };
