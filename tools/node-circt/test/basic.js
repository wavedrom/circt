'use strict';

const expect = require('chai').expect;
const lib = require('../lib/index.js');

describe('basic', () => {

  it('typeof circt', done => {
    expect(lib.circt).to.be.an('object');
    done();
  });

  it('typeof circt.hello', done => {
    expect(lib.circt.hello).to.be.an('function');
    done();
  });

});

/* eslint-env mocha */
