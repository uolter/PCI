import docclass

docclass.getwords('python is a dynamic language')

cl = docclass.naivebayes(docclass.getwords)

cl.setdb('test.db')

cl.train('pythons are constrictors', 'snake')
cl.train('python has dynamic types', 'language')
cl.train('python was developed as scripting language', 'language')

cl.classify('dynamic programming')

cl.classify('boa constrictors')
exit()
