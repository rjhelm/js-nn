const mmap = math.map; // to pass each elemnt of matrix to function
const rand = math.random; // to generate random number
const transp = math.transpose;
const mat = math.matrix;
const e = math.evaluate;
const sub = math.subtract;
const sqr = math.square;
const sum = math.sum;

// neural networks hyperparameters
const inputnodes = 784;
const hiddennodes = 100;
const outputnodes = 10;
const learningrate = 0.2;
const threshold = 0.5;
const iter = 0;
const iterations = 5;

/* path to the data sets */
const trainingDataPath = "./mnist/mnist_train.csv";
const testDataPath = "./mnist/mnist_test.csv";

/* these constants will be filled during data loading and preparation */
const trainingData = [];
const trainingLabels = [];
const testData = [];
const testLabels = [];

window.onload = async () => {
    myNN = new NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate);
    /* ... */
}
class NeuralNetwork {
    constructor(
        inputnodes,
        hiddennodes,
        outputnodes,
        learningrate,
        wih,
        who,
    ) {
        this.inputnodes = inputnodes;
        this.hiddennodes = hiddennodes;
        this.outputnodes = outputnodes;
        this.learningrate = learningrate;

        this.wih = wih || sub(mat(rand([hiddennodes, inputnodes])), 0.5);
        this.who = who || sub(mat(rand([outputnodes, hiddennodes])), 0.5);

        this.act = (matrix) => mmap(matrix, (x) => 1 / (1 + Math.exp(-x)));
    }
    cache = { loss: [] };

    static normalizeData = (data) => { /* ... */};

    forward = (input) => {
        const wih = this.wih;
        const who = this.who;
        const act = this.act;

        input = transp(mat([input]));

        const h_in = e("wih * input", { wih, input });
        const h_out = act(h_in);

        const o_in = e("who * h_out", { who, h_out });
        const actual = act(o_in);

        this.cache.input = input;
        this.cache.h_out = h_out;
        this.cache.actual = actual;

        return actual;
    };
    backward = (input, target) => {
        const who = this.who;
        const input = this.cache.input;
        const h_out = this.cache.h_out;
        const actual = this.cache.actual;

        target = transp(mat([target]));

        // calculate the gradient of error func (E) w.r.t. func (A)
        const dEdA = sub(target, actual);

        // calculate gradient of activation func (A) w.r.t. the sums ()
        const o_dAdZ = e("actual .* (1 - actual)", { actual });

        // calculate the error gradient of loss func w.r.t. the output of the network
        const dwho = e("(dEdA .* o_dAdZ) * h_out", {
            dEdA,
            o_dAdZ,
            h_out,
        });
        // calculate the weighted error for the hidden layer
        const h_err = e("who' * (dEdA .* o_dAdZ)", { who, dEdA, o_dAdZ });
        // calculate the gradient of activation func (A) w.r.t. the sums ()
        const h_dAdZ = e("h_out .* (1 - h_out), { h_out }");
        

        // calculate the error gradient of the loss function w.r.t. the input of the network
        const dwih = e("(h_err .* h_dAdZ) * input", {
            h_err,
            h_dAdZ,
            input,
        });

        this.cache.dwhi = dwih;
        this.cache.dwho = dwho;
        this.cache.loss.push(sum(sqr(dEdA)));
    };
    update = () => { 
        const wih = this.wih;
        const who = this.who;
        const dwih = this.cache.dwhi;
        this.dwho = this.cache.dwho;
        const r = this.learningrate;

        this.wih = e("wih + (r .* dwih)", { wih, dwih, r });
        this.who = e("who + (r .* dwho)", { who, dwho, r });
    };
    predict = (input) => {
        return this.forward(input);
    };
    train = (input, target) => {
        this.forward(input);
        this.backward(target);
        this.update();
    };
}