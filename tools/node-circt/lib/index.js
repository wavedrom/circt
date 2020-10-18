'use strict';

const binding = require('bindings');
const circt = binding('circt.node');

// for cmake
// const circt = binding('node_circt');


const pkg = require('../package.json');

module.exports = {
  version: pkg.version,
  circt
};
