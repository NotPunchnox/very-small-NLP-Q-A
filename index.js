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

      vocab.push(object)
    }
  }

}

console.log(vocab)
