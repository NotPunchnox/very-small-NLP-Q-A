/* Very Small SML */

// Training Data ( small dataset )
const text = [
  "Bonjour je suis un robot",
  "Un robot est une machine",
  "Je fonctionne avec un ordinateur"
];

const vocab = [];
const pairs = [];

// Make the tokenizer and build pairs
for (const line of text) {

  for (const word of line.split(' ')) {
    if (!Object.keys(vocab).includes(word)) {

      let object = { [vocab.length]: word }

      vocab.push(object);
      if (vocab.length % 2 == 0) {
        pairs.push([Object.keys(object)[0] - 1, Object.keys(object)[0]])
      }
    }
  }

}


// Function for tokenize text
function Tokenize(text) {
  const result = [];
  let text_parsed = text.split(' ');

  for (const word of text_parsed) {
    const token = Object.keys(vocab.find(a => a[Object.keys(a)[0]] == word))[0];
    result.push(token);
  }

  return result;
}

// Function for decode tokenized text
function Decode(tokens) {
  const result = [];

  for (const token of tokens) {
    const word = vocab[token][token];
    result.push(word);
  }

  return result;
}


// Tokenize text
const output = Tokenize("Bonjour je");
console.log(output);

// Decode tokenized text
const decoded_result = Decode(output);
console.log("decoded result:", decoded_result);


