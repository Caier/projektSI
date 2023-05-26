from c45tree import C45Tree

X = [
    ["Rainy", "Hot", 34, "False"],
    ["Rainy", "Hot", 35, "True"],
    ["Overcast", "Hot", 28, "False"],
    ["Sunny", "Mild", 27, "False"],
    ["Sunny", "Cool", 22, "False"],
    ["Sunny", "Cool", 21, "True"],
    ["Overcast", "Cool", 23, "True"],
    ["Rainy", "Mild", 27, "False"],
    ["Rainy", "Cool", 20, "False"],
    ["Sunny", "Mild", 23, "False"],
    ["Rainy", "Mild", 19, "True"],
    ["Overcast", "Mild", 20, "True"],
    ["Overcast", "Hot", 22, "False"],
    ["Sunny", "Mild", 30, "True"]
]

Y = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]

c45 = C45Tree(X, Y)
for x in X:
    print(c45.predict(x))
print(c45.get_tree_logic())