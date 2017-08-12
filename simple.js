var math = require('mathjs').create({ randomSeed: 'seed' });

var x = [
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
];
var y = [
    [0],
    [1],
    [0],
    [0]
];

var weights = math.random([3, 1], -1, 1);

for (var i = 0; i < 1000; i++) {
    var output = sigmoid(math.multiply(x, weights));

    var delta = math.multiply(
        math.transpose(x),
        math.dotMultiply(
            math.subtract(y, output),
            sigmoidPrime(output)));
    weights = math.add(weights, delta);
}

console.log(sigmoid(math.multiply([1, 0, 1], weights)));
console.log(sigmoid(math.multiply([1, 1, 0], weights)));


function sigmoid(x) {
    // 1 / (1 + exp(-x))
    return math.dotDivide(1, math.add(1, math.exp(math.dotMultiply(-1, x))));
}
function sigmoidPrime(sig) {
    // x * (1 - x)
    return math.dotMultiply(sig, math.subtract(1, sig))
}
