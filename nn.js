const mmap = math.map; // to pass each elemnt of matrix to function
const rand = math.random; // to generate random number
const transp = math.transpose;
const mat = math.matrix;
const e = math.evaluate;
const sub = math.subtract;
const sqr = math.square;
const sum = math.sum;

class NeuralNetwork {
    constructor(
        input_nodes,
        hidden_nodes,
        output_nodes,
        learning_rate,
        wih,
        who,
    ) {
        this.input_nodes = input_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;
        this.learning_rate = learning_rate;

        this.wih = wih || sub(mat(rand([hidden_nodes, input_nodes])), 0.5);
        this.who = who || sub(mat(rand([output_nodes, hidden_nodes])), 0.5);

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
    backward = (input, target) => { /* ... */};
    update = () => { /* ... */};
    predict = (input) => { /* ... */};
    train = (input, target) => { /* ... */};
}