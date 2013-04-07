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

### Strenghts and Weaknesses

- Speed at which it can be trained with large datasets
- Support for incremenal training: each new piece of training data can be used to update the probabilities without using any of the old training data
- Biggest downside: inability to deal with outcomes than change based on combinations of features.


## Decision Tree (Chapter 7)

- Extremely easy to understand and interpret
- It works based on the concept of **entropy** (the amount of disorder in a set). The entropy for each set is used to calculate the **information gain** defined as:

  ``
    p(i) = frequency(outcome) = count(outcome) / count(total rows)
    Entropy = sum(p(i) * log(p(i)) for all the outcomes

    weight1 = size of subset 1 / size of original set
    weight2 = size of subset 2 / size of original set

    gain = entropy(original) - weight1 * entropy(set1) - weight2 * entropy(set2)

  ``

### Code example

  
    import treepredict
   
    # fruits with their colors and size
    fruits = [
              [4, 'red', 'apple'],
              [4, 'green', 'apple'],
              [1, 'red', 'cherry'],
              [1, 'green', 'grape'],
              [5, 'red', 'apple']
             ]
    
    # train the tree
    tree = treepredict.buildtree(fruits)

    # some classification
    treepredict.classify([2, 'red'], tree)
    treepredict.classify([5, 'red'], tree)
    treepredict.classify([1, 'green'], tree)
    
    treepredict.printtree(tree)
    
    treepredict.drawtree(tree, jpeg='treeview.jpg')


  

this is the tree:

``
    0:4? 
    T-> {'apple': 3}
    F-> 1:green? 
      T-> {'grape': 1}
      F-> {'cherry': 1}
``


### Strenghts and Weaknesses

- useful not just for **classification**, but also for **interpretation**.
- ability to mix categorical and numerical data.
- it can easily cope with **interactions of variables**. This is an advantage over the Bayesian classifier
- It does not support incremental training.

