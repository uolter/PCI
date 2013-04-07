PCI
===

This is the example code from the book:

Programming Collective Intelligence By Toby Segaran. 
Copyright 2007 Toby Segaran, 978-0-596-52932-1


http://shop.oreilly.com/product/9780596529321.do



Bayesian Classifier (Chapter 6)
-------------------

- For documen classification system: spam filtering or dividing up a set of document 
- It works on any dataset that can be turned into list of features: A feature is something that is either present or absent fot a given item
- For documents the futures are the words in the document, but they could also be characteristics of an undefined object: symptons of a diseas etc

### naive Bayes classifier

  P(Category | Document) = P(Document | Categoty) * P(Category) / P(Document)
  
  where:
  
  P(Document | Category) = P(Word1 | Category) * P(Word2 | Cagetory) ........ = &prod;P(wi | Category)


### Code example

```
  import docclass

  docclass.getwords('python is a dynamic language')

  cl = docclass.naivebayes(docclass.getwords)

  cl.setdb('test.db')

  cl.train('pythons are constrictors', 'snake')
  cl.train('python has dynamic types', 'language')
  cl.train('python was developed as scripting language', 'language')

  cl.classify('dynamic programming')

  cl.classify('boa constrictors')
```  

### Strenghts and Weaknesss

- Speed at which it can be trained with large datasets
- Support for incremenal training: each new piece of training data can be used to update the probabilities without using any of the old training data
- Biggest downside: inability to deal with outcomes than change based on combinations of features.
