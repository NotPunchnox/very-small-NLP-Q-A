const text = [
  "Bonjour je suis un robot",
  "Un robot est une machine",
  "Je fonctionne avec un ordinateur"
];

const vocab = [];
const pairs = [];

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


function Tokenize(text) {
  const result = [];
  let text_parsed = text.split(' ');

  for (const word of text_parsed) {
    const token = Object.keys(vocab.find(a => a[Object.keys(a)[0]] == word))[0];
    result.push(token);
  }

  return result;
}


const output = Tokenize("Bonjour je");
console.log(output);
