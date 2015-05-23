import theano
import theano.tensor as TT
import time

import numpy

for n_hidden in [50, 100,500,1500]:
    for n_words in [10,54,3000]:
        theano.config.floatX='float32'
        data = numpy.random.randint(n_words, size=(100*100,))
        print data.shape

        h0_tm1 = TT.alloc(numpy.array(0, dtype=theano.config.floatX), n_hidden)
        rng = numpy.random.RandomState(123)
        floatX='float32'

        profile=0
        lr = .01
        W_uh = numpy.asarray(rng.normal(size=(n_words, n_hidden),
                                        loc=0.,
                                        scale=1./n_words),
                             dtype=floatX)

        W_hh = numpy.asarray(rng.normal(size=(n_hidden, n_hidden),
                                        loc=0.,
                                        scale=1./n_hidden),
                             dtype=floatX)
        W_hy = numpy.asarray(rng.normal(size=(n_hidden, n_words),
                                           loc=0.,
                                            scale=1./n_hidden),
                                dtype=floatX)
        b_y = numpy.zeros((n_words,), dtype=floatX)
        b_h = numpy.zeros((n_hidden,), dtype=floatX)

        W_uh = theano.shared(W_uh, 'W_uh')
        W_hh = theano.shared(W_hh, 'W_hh')
        b_h = theano.shared(b_h, name='b_h')
        W_hy = theano.shared(W_hy, 'W_hy')
        b_y = theano.shared(b_y,'b_y')


        t = TT.ivector('t')
        _T = TT.alloc(numpy.array(0, dtype=floatX), 100, n_words)
        arange = TT.constant(numpy.arange(100).astype('int32'))
        ones = TT.constant(numpy.ones((100,),
                                          dtype=floatX))
        T = TT.set_subtensor(_T[arange, t], ones)

        u = TT.ivector('u')
        U = TT.set_subtensor(_T[arange, u], ones)
        trans_inp = W_uh[u]
        trans_inp = TT.dot(U, W_uh)
        activ = TT.nnet.sigmoid


        def recurrent_fn(u_t, h_tm1, W_hh):
            h_t = activ(TT.dot(h_tm1, W_hh) + u_t )
            return h_t

        h, _ = theano.scan(
            recurrent_fn,
            sequences = trans_inp,
            outputs_info = h0_tm1,
            non_sequences = W_hh,
            name = 'recurrent_fn',
            truncate_gradient = -1,
            #mode = theano.Mode(linker='cvm'),
            profile=profile)
        y = TT.dot(h,W_hy)
        y = TT.nnet.softmax(y)
        # Reshape to a matrix due to limitation of advanced indexing
        cost = -TT.xlogx.xlogy0(T, y)
        cost = cost.sum(1).mean()
        updates = theano.compat.python2x.OrderedDict()
        # Compute gradients
        rval = TT.grad(cost, [W_hh, W_uh, W_hy])
        gW_hh, gW_uh, gW_hy = rval

        updates[W_hh] = W_hh - lr*gW_hh
        updates[W_uh] = W_uh - lr*gW_uh
        updates[W_hy] = W_hy - lr*gW_hy

        train_step = theano.function([u,t], [],
                                     #mode=theano.Mode(linker='cvm'),
                                     profile=profile,
                                     name='train_step',
                                     allow_input_downcast = True,
                                     updates=updates)

        st1 = time.time()
        for dx in xrange(100):
            train_step(data[dx*100:dx*100+100], data[dx*100:dx*100+100])
        ed1 = time.time()
        st2 = time.time()
        for dx in xrange(100):
            train_step(data[dx*100:dx*100+100], data[dx*100:dx*100+100])
        ed2 = time.time()
        print 'For n_words, ', n_words, ' n_hidden', n_hidden, 'it took',
        print 10000./(ed1-st1),
        print 10000./(ed2-st2)
