get_ipython().magic(u'logstart example.py append')
import nn
online, pharmacy = 1, 2
spam, notspam = 1, 2
possible = [spam, notspam]
neuralnet = nn.searchnet('nntest.db')
neuralnet.maketables()
neuralnet.trainquery([online], possible, notspam)
neuralnet.trainquery([online, pharmacy], possible, spam)
neuralnet.trainquery([pharmacy], possible, notspam)
neuralnet.getresult([online, pharmacy], possible)
neuralnet.getresult([online], possible)
neuralnet.trainquery([online], possible, notspam)
neuralnet.getresult([online], possible)
neuralnet.trainquery([online], possible, notspam)
neuralnet.getresult([online], possible)
quit()
