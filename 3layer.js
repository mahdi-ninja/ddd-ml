var math = require('mathjs').create({ randomSeed: 'seed' });

var input_dataset = [
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
];
var output_dataset = [
    [0],
    [1],
    [1],
    [0]
];

//initialise weights matrix with values in (-1, 1)
var weights0 = math.random([3, 4], -1, 1);
var weights1 = math.random([4, 1], -1, 1);

for (var i = 0; i < 60000; i++) {

    //forward propagation
    var l0 = input_dataset;
    var l1 = sigmoid(math.multiply(l0, weights0));
    var l2 = sigmoid(math.multiply(l1, weights1));

    var l2_err = math.subtract(output_dataset, l2);
    var l2_delta = math.dotMultiply(l2_err, sigmoidPrime(l2));
    var l1_err = math.multiply(l2_delta, math.transpose(weights1));
    var l1_delta = math.dotMultiply(l1_err, sigmoidPrime(l1));

    weights1 = math.add(weights1, math.multiply(math.transpose(l1), l2_delta));
    weights0 = math.add(weights0, math.multiply(math.transpose(l0), l1_delta));

    if (i % 1000 == 0) {
        console.log('err: ' + math.mean(math.abs(l2_err)));
    }
}

var l1 = sigmoid(math.multiply([1, 1, 0], weights0));
var l2 = sigmoid(math.multiply(l1, weights1));
console.log(l2);


function sigmoid(x) {
    // 1 / (1 + exp(-x))
    return math.dotDivide(1, math.add(1, math.exp(math.dotMultiply(-1, x))));
}
function sigmoidPrime(sig) {
    // x * (1 - x)
    return math.dotMultiply(sig, math.subtract(1, sig))
}
