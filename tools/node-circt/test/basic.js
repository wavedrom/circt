'use strict';

const expect = require('chai').expect;
const lib = require('../lib/index.js');
console.log(lib);

describe('basic', () => {

  it('typeof circt', () =>
    expect(lib.circt).to.be.an('object'));

  it('typeof circt.hello', () =>
    expect(lib.circt.hello).to.be.an('function'));

  it('typeof circt.getNewContext', () =>
    expect(lib.circt.getNewContext).to.be.an('function'));

  it('getNewContext', () =>
    expect(lib.circt.getNewContext()).to.be.an('object'));

});

/* eslint-env mocha */
