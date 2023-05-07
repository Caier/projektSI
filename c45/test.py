from c45tree import C45Tree

X = [
    ["Rainy", "Hot", "High", "False"],
    ["Rainy", "Hot", "High", "True"],
    ["Overcast", "Hot", "High", "False"],
    ["Sunny", "Mild", "High", "False"],
    ["Sunny", "Cool", "Normal", "False"],
    ["Sunny", "Cool", "Normal", "True"],
    ["Overcast", "Cool", "Normal", "True"],
    ["Rainy", "Mild", "High", "False"],
    ["Rainy", "Cool", "Normal", "False"],
    ["Sunny", "Mild", "Normal", "False"],
    ["Rainy", "Mild", "Normal", "True"],
    ["Overcast", "Mild", "High", "True"],
    ["Overcast", "Hot", "Normal", "False"],
    ["Sunny", "Mild", "High", "True"]
]

Y = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]

c45 = C45Tree(X, Y)
for row in X:
    print(c45.predict(row))