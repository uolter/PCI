import treepredict
# fruits with their colors and size
fruits = [
[4, 'red', 'apple'],
[4, 'green', 'apple'],
[1, 'red', 'cherry'],
[1, 'green', 'grape'],
[5, 'red', 'apple']
]
tree = treepredict.buildtree(fruits)
treepredict.classify([2, 'red'], tree)
treepredict.classify([5, 'red'], tree)
treepredict.classify([1, 'green'], tree)
treepredict.printtree(tree)
#treepredict.drawtree(tree, jpeg='treeview.jpg')
