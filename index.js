const { Tokenizer } = require('./src/tokenizer.js');

const tokenizer = new Tokenizer();

tokenizer.init();

// Tokenize example text
const tokens = tokenizer.tokenize("Bonjour je suis samuel");


