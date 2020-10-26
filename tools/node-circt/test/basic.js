'use strict';

const expect = require('chai').expect;
const {circt} = require('../lib/index.js');

describe('basic', () => {

  it('typeof circt', () =>
    expect(circt).to.be.an('object'));

  it('typeof circt.getNewContext', () =>
    expect(circt.getNewContext).to.be.an('function'));

  it('typeof circt.toStringMLIR', () =>
    expect(circt.toStringMLIR).to.be.an('function'));

  it('circt.toStringMLIR()', () => {
    const cxt = circt.getNewContext();
    expect(cxt).to.be.an('object');
    expect(circt.toStringMLIR(cxt)).to.eq('world');
  });

});

/* eslint-env mocha */
