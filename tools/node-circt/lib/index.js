'use strict';

// const binding = require('node-gyp-build')(__dirname)
const binding = require('bindings');

const pkg = require('../package.json');

const circt = binding('circt.node');

module.exports = {
  version: pkg.version,
  circt
};
